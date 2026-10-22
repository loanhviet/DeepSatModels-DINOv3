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
from datetime import datetime, timezone
from importlib.metadata import version

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from data import get_dataloaders, get_loss_data_input
from metrics.loss_functions import get_loss
from models import get_model
from models.TSViT.TSViTDINOv3 import TSViTDINOv3
from models.TSViT.TSViTdense import TSViT, Transformer
from models.TSViT.module import Attention
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_net_trainable_params, load_from_checkpoint


def backbone_digest(backbone):
    digest = hashlib.sha256()
    for value in backbone.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml')
    parser.add_argument('--output', help='Output directory (default depends on architecture)')
    args = parser.parse_args()
    config = read_yaml(args.config)
    temporal_only = config['MODEL']['architecture'] == 'TSViT_DINOv3_Temporal'
    output = Path(args.output or ('outputs/dinov3-temporal-integration' if temporal_only
                                 else 'outputs/dinov3-integration'))
    output.mkdir(parents=True, exist_ok=True)
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
    original = TSViT({**config['MODEL'], 'spatial_depth': 4}).to(device)
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
    if temporal_only:
        assert not hasattr(net, 'space_transformer')
        assert not hasattr(net, 'space_pos_embedding')
        assert not any(name.startswith('space_') for name in net.state_dict())
        transformers = [net.temporal_transformer]
    else:
        assert type(net).forward is TSViT.forward
        transformers = [net.temporal_transformer, net.space_transformer]
    for transformer in transformers:
        assert type(transformer) is Transformer
        assert all(type(layer[0].fn) is Attention for layer in transformer.layers)
    net.train()
    assert not any(module.training for module in adapter.backbone.modules())
    assert not any(p.requires_grad for p in adapter.backbone.parameters())
    pretrained_path = Path(config['MODEL']['dinov3_checkpoint']).expanduser() / 'model.safetensors'
    pretrained = load_file(str(pretrained_path))
    loaded = adapter.backbone.state_dict()
    assert loaded.keys() == pretrained.keys()
    for name, value in loaded.items():
        assert torch.equal(value.detach().cpu(), pretrained[name]), name
    pretrained_tensor_count = len(pretrained)
    del pretrained, loaded
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
    optimizer = torch.optim.AdamW(get_net_trainable_params(net), lr=float(config['SOLVER']['lr_base']),
                                 weight_decay=float(config['SOLVER'].get('weight_decay', 0)))
    optimizer_params = [p for group in optimizer.param_groups for p in group['params']]
    expected_ids = {id(p) for p in net.parameters() if p.requires_grad}
    assert {id(p) for p in optimizer_params} == expected_ids
    assert len(optimizer_params) == len(expected_ids)
    optimizer.zero_grad()
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
    labels, mask = ground_truth
    valid = mask.squeeze(-1)
    assert torch.equal(mask, labels != 19)
    reference_loss = F.cross_entropy(logits.permute(0, 2, 3, 1)[valid],
                                     labels.squeeze(-1)[valid].long())
    torch.testing.assert_close(loss, reference_loss)
    actual_gradient, = torch.autograd.grad(loss, logits, retain_graph=True)
    expected_gradient, = torch.autograd.grad(reference_loss, logits, retain_graph=True)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    void_gradient = actual_gradient.permute(0, 2, 3, 1)[~valid]
    assert void_gradient.numel() > 0 and torch.count_nonzero(void_gradient) == 0
    void_gradient_max = void_gradient.abs().max().item()
    del reference_loss, actual_gradient, expected_gradient, void_gradient
    loss.backward()
    gradient_norms = {}
    gradient_l2 = {}
    trainable_groups = ['to_patch_embedding.projection', 'to_temporal_embedding_input',
                        'temporal_transformer', 'mlp_head', 'temporal_token']
    if not temporal_only:
        trainable_groups.extend(['space_transformer', 'space_pos_embedding'])
    for prefix in trainable_groups:
        gradients = [p.grad for name, p in net.named_parameters() if name.startswith(prefix)]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        gradient_norms[prefix] = sum(g.norm().item() for g in gradients)
        assert gradient_norms[prefix] > 0
        gradient_l2[prefix] = sum(g.double().square().sum().item() for g in gradients) ** 0.5
    assert all(p.grad is None for p in adapter.backbone.parameters())
    assert all(torch.equal(before[name], p) for name, p in net.named_parameters() if name in before)
    optimizer.step()
    parameter_updates = {}
    for prefix in gradient_norms:
        deltas = [(p.detach() - before[name]).double() for name, p in net.named_parameters()
                  if name in before and name.startswith(prefix)]
        assert all(torch.isfinite(delta).all() for delta in deltas)
        changed = sum(bool(torch.count_nonzero(delta)) for delta in deltas)
        parameter_updates[prefix] = {
            'delta_l2': sum(delta.square().sum().item() for delta in deltas) ** 0.5,
            'delta_max_abs': max(delta.abs().max().item() for delta in deltas),
            'changed_tensors': changed,
            'total_tensors': len(deltas),
        }
        assert changed > 0
    digest_after = backbone_digest(adapter.backbone)
    assert digest_before == digest_after
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
    if temporal_only:
        # Independently reuse the ORIGINAL TSViT.forward with its spatial
        # block bypassed, to verify token layout and head reconstruction.
        original_forward = TSViTDINOv3({**config['MODEL'], 'spatial_depth': 0}).to(device)
        original_forward.space_transformer = torch.nn.Identity()
        with torch.no_grad():
            original_forward.space_pos_embedding.zero_()
        missing, unexpected = original_forward.load_state_dict(net.state_dict(), strict=False)
        assert missing == ['space_pos_embedding'] and not unexpected
        original_forward.eval()
        with torch.no_grad():
            torch.testing.assert_close(reference, original_forward(inputs), atol=0, rtol=0)
            # Different pairs of real dates catch B/class/patch mixing that
            # a batch-size-one test cannot detect.
            two_samples = torch.cat((inputs[:, :2], inputs[:, 1:3]), dim=0)
            torch.testing.assert_close(net(two_samples), original_forward(two_samples),
                                       atol=0, rtol=0)
        del two_samples
        del original_forward
    checkpoint = output / 'roundtrip.pth'
    torch.save(net.state_dict(), checkpoint)
    restored = get_model(config, device)
    load_from_checkpoint(restored, str(checkpoint), device=device)
    restored.eval()
    with torch.no_grad():
        reloaded = restored(inputs)
    torch.testing.assert_close(reference, reloaded, atol=0, rtol=0)

    report = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in ['models/TSViT/TSViTDINOv3.py', 'metrics/loss_functions.py',
                         'tests/smoke_dinov3.py', 'utils/torch_utils.py',
                         'train_and_eval/segmentation_training_transf.py']
        },
        'config': args.config,
        'architecture': config['MODEL']['architecture'],
        'seed': config['SOLVER']['seed'],
        'python': platform.python_version(),
        'transformers': version('transformers'),
        'sample_path': str(Path(dataset.root_dir) / dataset.data_paths.iloc[0, 0]),
        'sample_sha256': hashlib.sha256((Path(dataset.root_dir) / dataset.data_paths.iloc[0, 0]).read_bytes()).hexdigest(),
        'device': torch.cuda.get_device_name(device),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'input_shape': list(inputs.shape), 'logits_shape': list(logits.shape),
        'real_frames': int((~padded).sum()), 'padding_frames': int(padded.sum()),
        'valid_label_pixels': int(ground_truth[1].sum()),
        'loss': loss.item(), 'rgb_max_abs_error': rgb_max_error,
        'gradient_norms': gradient_norms,
        'gradient_norms_definition': 'Legacy sum of per-tensor L2 gradient norms',
        'gradient_l2': gradient_l2,
        'gradient_l2_definition': 'L2 over all gradient elements in each group; float64 accumulation',
        'parameter_updates': parameter_updates,
        'weights_unchanged_after_backward_before_step': True,
        'optimizer': {
            'class': type(optimizer).__name__,
            'parameter_helper': 'utils.torch_utils.get_net_trainable_params',
            'lr': optimizer.param_groups[0]['lr'],
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
            'betas': list(optimizer.param_groups[0]['betas']),
            'eps': optimizer.param_groups[0]['eps'],
            'trainable_parameter_tensors': len(expected_ids),
            'optimizer_parameter_tensors': len(optimizer_params),
            'exact_trainable_coverage_no_duplicates': True,
        },
        'loss_and_logits_gradient_match_valid_pixel_cross_entropy': True,
        'void_logits_gradient_max_abs': void_gradient_max,
        'pretrained_checkpoint': str(pretrained_path),
        'pretrained_checkpoint_sha256': hashlib.sha256(pretrained_path.read_bytes()).hexdigest(),
        'pretrained_tensors_exactly_loaded': pretrained_tensor_count,
        'backbone_sha256_before': digest_before,
        'backbone_sha256_after': digest_after,
        'backbone_sha256_before_and_after': digest_before,
        'backbone_unchanged': True,
        'inherited_original_forward': not temporal_only,
        'original_temporal_attention': True,
        'spatial_encoder_removed': temporal_only,
        'bypassed_original_forward_exact': temporal_only,
        'batch_size_two_forward_exact': temporal_only,
        'trainable_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad),
        'frozen_parameters': sum(p.numel() for p in net.parameters() if not p.requires_grad),
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
