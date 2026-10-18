# DeepSatModels — DINOv3 for Satellite Image Time Series

This repository extends [DeepSatModels](https://github.com/michaeltrs/DeepSatModels)
from commit `53e655936be088522ec419b3ead38684c20ae1ec`. It integrates DINOv3
with TSViT for crop-type semantic segmentation on PASTIS24 while reusing the
original data, training, loss, and evaluation pipeline.

Local runs are used only to check that the implementation works. Full
training and evaluation on separate dataset splits have not been performed.

## Architectures

| `MODEL.architecture` | Processing pipeline | Change from the original TSViT |
| --- | --- | --- |
| `TSViT` | Patch embedding → Temporal TSViT → Spatial TSViT → head | Original model |
| `TSViT_DINOv3` | DINOv3 → Projection → Temporal TSViT → Spatial TSViT → head | Replaces patch embedding with DINOv3 features |
| `TSViT_DINOv3_Temporal` | DINOv3 → Projection → Temporal TSViT → head | Replaces patch embedding and removes the TSViT spatial Transformer |

Both DINOv3 variants use RGB input and a frozen backbone. The projection,
temporal Transformer, and segmentation head remain trainable. The variant
that retains the spatial Transformer also trains that block. Both use the
original masked cross-entropy loss.

The temporal variant retains the TSViT temporal encoder; it does not replace
the entire TSViT encoder with DINOv3.

## Installation

The following environment is for local smoke checks. Run commands from the
repository root. The tested environment is Linux,
Python 3.13.14, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, and an NVIDIA
RTX 3050 Laptop GPU with 4 GB VRAM. Install `uv` and an NVIDIA driver
compatible with CUDA 12.8 before creating the environment.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-dinov3.txt
```

The legacy conda environment in `deepsatmodels_env.yml` is retained as part
of the original repository. Use the separate environment above for DINOv3.

## Setup and execution

1. Prepare a DINOv3 checkpoint in Hugging Face format and PASTIS24 pickle data.
2. Select a YAML configuration and set its checkpoint and data paths.
3. Configure the YAML paths and run the real-sample checks.

[README_DINOV3.md](README_DINOV3.md) describes the data format, configuration
fields, test and training commands, and code changes. The `*_smoke.yaml`
files currently reference a sample on the machine used for validation;
update these paths on another machine. The `*_fold1.yaml` files contain
`/path/to/PASTIS24` placeholders that must be replaced before use.

For example, after configuring the paths for the temporal variant:

```bash
.venv/bin/python tests/smoke_dinov3.py \
  --config configs/PASTIS24/TSViT_DINOv3_Temporal_small_smoke.yaml
.venv/bin/python train_and_eval/segmentation_training_transf.py \
  --config configs/PASTIS24/TSViT_DINOv3_Temporal_small_smoke.yaml --device 0
```

## Local validation status

DINOv3 Small has passed forward/backward checks, one AdamW update,
training/evaluation through the original entrypoint, and checkpoint
save/load checks using the real sample `10000_0.pickle`.

| Architecture | Real-sample smoke checks |
| --- | --- |
| `TSViT_DINOv3` | Passed |
| `TSViT_DINOv3_Temporal` | Passed |

Conditions: batch size 1, T=60, RGB, and a frozen backbone. The local checks
use the same sample for a training step and an evaluation pass. They verify
execution and checkpoint handling only. No segmentation quality results
are available yet. Full PASTIS24 training, separate evaluation splits, and
the Base/Satellite checkpoints have not been evaluated.

Detailed logs, checksums, and memory diagnostics are retained in the
validation reports:

- [Patch embedding replacement](docs/dinov3-smoke.md).
- [DINOv3 with the TSViT temporal encoder](docs/dinov3-temporal-smoke.md).

## Documentation and references

- [DINOv3 integration and usage guide](README_DINOV3.md).
- [Original TSViT documentation](README_TSVIT.md), [Context-self contrastive learning](README_CSCL.md).
- [ViTs for SITS — Tarasiou et al., CVPR 2023](https://arxiv.org/abs/2301.04944).
- [Context-self contrastive pretraining — Tarasiou et al., TGRS 2022](https://ieeexplore.ieee.org/abstract/document/9854891).
- [DINOv3 source code and models](https://github.com/facebookresearch/dinov3).

DeepSatModels source code is distributed under the [Apache License 2.0](LICENSE.txt).
DINOv3 weights are subject to the license included with each checkpoint.

Copyright © 2023 Michail Tarasiou.
