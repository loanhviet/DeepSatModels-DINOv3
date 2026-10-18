"""DINOv3 patch features feeding the unchanged temporal/spatial TSViT."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from transformers import AutoModel

from data.PASTIS24.data_transforms import Normalize
from models.TSViT.TSViTdense import TSViT


class DINOv3PatchEmbedding(nn.Module):
    """Accept BTCHW normalized PASTIS bands; return (B*h*w), T, dim.

    All-zero spectral frames are the original transform's padding sentinel.
    Their DINO features are zero BEFORE projection. AFTER projection they
    equal the learned projection bias, just as the original Linear embeds
    a zero patch as its bias. Temporal attention is deliberately unchanged.
    """

    def __init__(self, model_config):
        super().__init__()
        if model_config['num_channels'] != 11:
            raise ValueError('DINOv3 adapter expects PASTIS: 10 bands + DOY')
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        if self.image_size % self.patch_size:
            raise ValueError('img_res must be divisible by TSViT patch_size')
        self.grid_size = self.image_size // self.patch_size
        self.frame_chunk_size = model_config.get('frame_chunk_size', 1)
        if self.frame_chunk_size < 1:
            raise ValueError('frame_chunk_size must be positive')
        checkpoint = Path(model_config['dinov3_checkpoint']).expanduser()
        if not checkpoint.is_dir():
            raise FileNotFoundError('Local DINOv3 checkpoint not found: %s' % checkpoint)
        self.backbone = AutoModel.from_pretrained(str(checkpoint), local_files_only=True)
        if self.backbone.config.model_type != 'dinov3_vit':
            raise ValueError('Expected a Hugging Face DINOv3 ViT checkpoint')
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.dino_image_size = self.grid_size * self.backbone.config.patch_size
        self.projection = nn.Linear(self.backbone.config.hidden_size, model_config['dim'])

        # Reuse the exact statistics used by the original data pipeline.
        normalizer = Normalize()
        rgb_indices = [2, 1, 0]  # B04, B03, B02 in PASTIS S2 band order
        self.register_buffer('rgb_indices', torch.tensor(rgb_indices))
        for name, values in [('pastis_mean', normalizer.mean_fold1),
                             ('pastis_std', normalizer.std_fold1)]:
            self.register_buffer(name, torch.tensor(values[:, rgb_indices].copy()))
        with (checkpoint / 'preprocessor_config.json').open() as handle:
            preprocessing = json.load(handle)
        for name, key in [('dino_mean', 'image_mean'), ('dino_std', 'image_std')]:
            self.register_buffer(name, torch.tensor(preprocessing[key]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        # Parent net.train() must not enable DINO's positional augmentation.
        self.backbone.eval()
        return self

    def prepare_rgb(self, frames):
        rgb = frames.index_select(1, self.rgb_indices)
        rgb = (rgb * self.pastis_std + self.pastis_mean) / 10000.0
        rgb = rgb.clamp(0, 1)
        rgb = F.interpolate(rgb, size=(self.dino_image_size, self.dino_image_size),
                            mode='bilinear', align_corners=False, antialias=True)
        return (rgb - self.dino_mean) / self.dino_std

    def forward(self, x):
        if x.ndim != 5 or x.shape[2:] != (10, self.image_size, self.image_size):
            raise ValueError('Expected [B,T,10,img_res,img_res] normalized PASTIS input')
        b, t = x.shape[:2]
        frames = x.reshape(b * t, 10, self.image_size, self.image_size)
        valid_indices = frames.ne(0).any(dim=(1, 2, 3)).nonzero(as_tuple=True)[0]
        num_patches = self.grid_size ** 2
        with torch.no_grad():
            features = frames.new_zeros((b * t, num_patches, self.backbone.config.hidden_size))
            for start in range(0, valid_indices.numel(), self.frame_chunk_size):
                indices = valid_indices[start:start + self.frame_chunk_size]
                rgb = self.prepare_rgb(frames.index_select(0, indices))
                tokens = self.backbone(pixel_values=rgb).last_hidden_state
                patch_tokens = tokens[:, 1 + self.backbone.config.num_register_tokens:]
                if patch_tokens.shape[1] != num_patches:
                    raise RuntimeError('DINOv3 token grid does not match TSViT grid')
                features.index_copy_(0, indices, patch_tokens.to(features.dtype))
        # Keep the bias for padded frames; do not zero the projected embeddings
        # or introduce a new temporal attention mask.
        embeddings = self.projection(features)
        return rearrange(embeddings.reshape(b, t, num_patches, -1),
                         'b t n d -> (b n) t d')


class TSViTDINOv3(TSViT):
    """Replace patch embedding only; inherit TSViT.forward and attention."""

    def __init__(self, model_config):
        super().__init__(model_config)
        self.to_patch_embedding = DINOv3PatchEmbedding(model_config)

    @property
    def trainable_params(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
