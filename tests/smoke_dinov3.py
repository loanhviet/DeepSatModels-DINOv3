"""Real-data integration checks; all paths come from the experiment YAML.

Run from the repository root with .venv/bin/python tests/smoke_dinov3.py.
This checks mechanics, not generalization or benchmark accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from data import get_dataloaders, get_loss_data_input
from metrics.loss_functions import get_loss
from models import get_model
from models.TSViT.TSViTdense import TSViT, Transformer
from models.TSViT.module import Attention
from utils.config_files_utils import read_yaml
from utils.torch_utils import load_from_checkpoint


def backbone_digest(backbone):
    digest = hashlib.sha256()
    for value in backbone.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml')
    parser.add_argument('--output', default='outputs/dinov3-integration')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = read_yaml(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this VRAM smoke check')
    device = torch.device('cuda:0')
    torch.manual_seed(config['SOLVER']['seed'])
    dataloaders = get_dataloaders(config)
    sample = next(iter(dataloaders['train']))
    inputs = sample['inputs'].to(device)
    ground_truth = get_loss_data_input(config)(sample, device)
    criterion = get_loss(config, device)

    # Regress the original model with the same unmodified data and loss.
    original = TSViT(config['MODEL']).to(device)
    original_logits = original(inputs)
    original_loss = criterion(original_logits.permute(0, 2, 3, 1), ground_truth)
    assert torch.isfinite(original_loss)
    original_loss.backward()
    assert original.to_patch_embedding[1].weight.grad.abs().sum() > 0
    original_result = {'logits_shape': list(original_logits.shape), 'loss': original_loss.item()}
    del original, original_logits, original_loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()

    torch.manual_seed(config['SOLVER']['seed'])
    net = get_model(config, device)
    adapter = net.to_patch_embedding
    assert type(net).forward is TSViT.forward
    for transformer in (net.temporal_transformer, net.space_transformer):
        assert type(transformer) is Transformer
        assert all(type(layer[0].fn) is Attention for layer in transformer.layers)
    net.train()
    assert not adapter.backbone.training
    assert not any(p.requires_grad for p in adapter.backbone.parameters())
    digest_before = backbone_digest(adapter.backbone)

    # An entirely padded chunk must skip DINO and retain projection bias.
    all_padding = inputs.new_zeros((1, 2, 10, 24, 24))
    padding_embedding = adapter(all_padding)
    torch.testing.assert_close(padding_embedding,
                               adapter.projection.bias.expand_as(padding_embedding))
    del all_padding, padding_embedding

    # Compare RGB preprocessing against raw pixels, not an inverse of itself.
    dataset = dataloaders['train'].dataset
    raw = dataset.read(0)
    rgb = adapter.prepare_rgb(inputs[:, 0, ..., :-1].permute(0, 3, 1, 2))
    expected = torch.tensor(raw['img'][0, [2, 1, 0]].copy(), device=device,
                            dtype=torch.float32).unsqueeze(0).div(10000).clamp(0, 1)
    expected = F.interpolate(expected, size=(192, 192), mode='bilinear',
                             align_corners=False, antialias=True)
    expected = (expected - adapter.dino_mean) / adapter.dino_std
    torch.testing.assert_close(rgb, expected, atol=2e-6, rtol=1e-5)
    rgb_max_error = (rgb - expected).abs().max().item()
    del rgb, expected

    # Verify B/T/patch ordering and chunk independence against direct DINO
    # extraction, using two samples with reversed frame order.
    frames = inputs[0, :2, ..., :-1].permute(0, 3, 1, 2)
    ordered = torch.stack((frames, frames.flip(0)))
    with torch.no_grad():
        chunk1 = adapter(ordered)
        adapter.frame_chunk_size = 2
        chunk2 = adapter(ordered)
        adapter.frame_chunk_size = config['MODEL']['frame_chunk_size']
        tokens = adapter.backbone(pixel_values=adapter.prepare_rgb(
            ordered.reshape(4, 10, 24, 24))).last_hidden_state
        direct = adapter.projection(tokens[:, 1 + adapter.backbone.config.num_register_tokens:])
        direct = direct.reshape(2, 2, 144, 128).permute(0, 2, 1, 3).reshape(288, 2, 128)
    torch.testing.assert_close(chunk1, direct, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(chunk1, chunk2, atol=1e-5, rtol=1e-4)
    del frames, ordered, chunk1, chunk2, tokens, direct

    captured = {}
    hook = adapter.register_forward_hook(
        lambda module, arguments, result: captured.update(embedding=result.detach().clone()))
    optimizer = torch.optim.AdamW(net.trainable_params, lr=float(config['SOLVER']['lr_base']),
                                 weight_decay=float(config['SOLVER'].get('weight_decay', 0)))
    before = {name: parameter.detach().clone() for name, parameter in net.named_parameters()
              if parameter.requires_grad}
    logits = net(inputs)
    hook.remove()
    assert list(logits.shape) == [inputs.shape[0], 19, 24, 24]
    assert captured['embedding'].shape == (inputs.shape[0] * 144, 60, 128)

    # The real sample has 43 observations + 17 padded frames. Projection bias
    # is retained, followed by the original DOY=0 embedding in TSViT.forward.
    padded = inputs.eq(0).all(dim=(2, 3, 4))
    assert padded.any()
    embeddings = captured.pop('embedding').reshape(inputs.shape[0], 144, 60, 128)
    for batch in range(inputs.shape[0]):
        actual = embeddings[batch, :, padded[batch]]
        torch.testing.assert_close(actual, adapter.projection.bias.expand_as(actual))
    del embeddings

    loss = criterion(logits.permute(0, 2, 3, 1), ground_truth)
    assert torch.isfinite(loss)
    loss.backward()
    gradient_norms = {}
    for prefix in ('to_patch_embedding.projection', 'to_temporal_embedding_input',
                   'temporal_transformer', 'space_transformer', 'mlp_head'):
        gradients = [p.grad for name, p in net.named_parameters() if name.startswith(prefix)]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        gradient_norms[prefix] = sum(g.norm().item() for g in gradients)
        assert gradient_norms[prefix] > 0
    assert all(p.grad is None for p in adapter.backbone.parameters())
    optimizer.step()
    for prefix in gradient_norms:
        assert any(not torch.equal(before[name], p) for name, p in net.named_parameters()
                   if name.startswith(prefix) and name in before)
    assert digest_before == backbone_digest(adapter.backbone)
    assert not adapter.backbone.training
    torch.cuda.synchronize(device)
    forward_backward_profile = {
        'elapsed_seconds': time.perf_counter() - started_at,
        'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
    }

    # Full state_dict remains compatible with the repository checkpoint loader.
    net.eval()
    with torch.no_grad():
        reference = net(inputs)
    checkpoint = output / 'roundtrip.pth'
    torch.save(net.state_dict(), checkpoint)
    restored = get_model(config, device)
    load_from_checkpoint(restored, str(checkpoint), device=device)
    restored.eval()
    with torch.no_grad():
        reloaded = restored(inputs)
    torch.testing.assert_close(reference, reloaded, atol=0, rtol=0)

    report = {
        'config': args.config,
        'seed': config['SOLVER']['seed'],
        'python': platform.python_version(),
        'sample_path': str(Path(dataset.root_dir) / dataset.data_paths.iloc[0, 0]),
        'sample_sha256': hashlib.sha256((Path(dataset.root_dir) / dataset.data_paths.iloc[0, 0]).read_bytes()).hexdigest(),
        'device': torch.cuda.get_device_name(device),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'input_shape': list(inputs.shape), 'logits_shape': list(logits.shape),
        'real_frames': int((~padded).sum()), 'padding_frames': int(padded.sum()),
        'valid_label_pixels': int(ground_truth[1].sum()),
        'loss': loss.item(), 'rgb_max_abs_error': rgb_max_error,
        'gradient_norms': gradient_norms,
        'backbone_sha256_before_and_after': digest_before,
        'backbone_unchanged': True, 'original_forward_and_attention': True,
        'padding_after_projection': 'projection.bias (no new attention mask)',
        'checkpoint_roundtrip_exact': True,
        'batch_time_patch_order_and_chunking': True,
        'original_tsvit': original_result,
        'forward_backward_profile': forward_backward_profile,
        'checkpoint_path': str(checkpoint),
    }
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
