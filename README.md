# DeepSatModels — DINOv3 Backbone for TSViT (PASTIS24)

This repository extends [DeepSatModels](https://github.com/michaeltrs/DeepSatModels) from commit `53e655936be088522ec419b3ead38684c20ae1ec`. It replaces TSViT’s spatial patch embedding with a **frozen DINOv3** ViT backbone for crop-type semantic segmentation on PASTIS24, while reusing the original data loading, training script, masked cross-entropy loss, and evaluation pipeline.

Local runs validate the integration on a small real sample. Full-dataset training and metrics on held-out splits have not been reported yet.

---

## What changed

### Scope

| Term | Implementation |
| --- | --- |
| Backbone / spatial encoder | `DINOv3PatchEmbedding` in `models/TSViT/TSViTDINOv3.py` |
| Replaced module | TSViT `to_patch_embedding` (linear embed on 10 spectral bands) |
| Preserved | Temporal (and optionally spatial) TSViT Transformers, DOY embedding, segmentation head, PASTIS24 pipeline |

```text
Original TSViT:
  images → Patch embedding → Temporal TSViT → Spatial TSViT → Head

TSViT_DINOv3:
  images → RGB adapter → DINOv3 (frozen) → Projection → Temporal → Spatial → Head

TSViT_DINOv3_Temporal:
  images → RGB adapter → DINOv3 → Projection → Temporal → Head
  (TSViT Spatial Transformer removed)
```

The temporal variant still uses the TSViT temporal encoder; DINOv3 does not replace the full TSViT stack.

### Modified and new files

| Priority | Path | Description |
| --- | --- | --- |
| Core | [`models/TSViT/TSViTDINOv3.py`](models/TSViT/TSViTDINOv3.py) | DINOv3 adapter, projection, `TSViT_DINOv3`, `TSViT_DINOv3_Temporal` |
| Registration | [`models/__init__.py`](models/__init__.py) | Architecture names in YAML; lazy imports for non-TSViT models |
| Config | [`configs/PASTIS24/TSViT_DINOv3_*.yaml`](configs/PASTIS24/), [`TSViT_DINOv3_Temporal_*.yaml`](configs/PASTIS24/) | Smoke and fold1 experiment configs |
| Tests | [`tests/smoke_dinov3.py`](tests/smoke_dinov3.py), [`tests/fixtures/pastis24-smoke.csv`](tests/fixtures/pastis24-smoke.csv) | Integration smoke on a real pickle sample |
| Dependencies | [`requirements-dinov3.txt`](requirements-dinov3.txt) | PyTorch + Transformers stack for DINOv3 |
| Training | [`train_and_eval/segmentation_training_transf.py`](train_and_eval/segmentation_training_transf.py) | Optional seed, CUDA memory logging, SummaryWriter cleanup |
| Data / utils | [`data/__init__.py`](data/__init__.py), [`data/datasets.yaml`](data/datasets.yaml), [`utils/config_files_utils.py`](utils/config_files_utils.py) | Configurable dataset paths; YAML safe load |
| Documentation | [`README_DINOV3.md`](README_DINOV3.md), [`docs/dinov3-smoke.md`](docs/dinov3-smoke.md), [`docs/dinov3-temporal-smoke.md`](docs/dinov3-temporal-smoke.md), [`docs/validation/`](docs/validation/) | Architecture details, smoke reports, provenance |

Unchanged for the core TSViT design: [`models/TSViT/TSViTdense.py`](models/TSViT/TSViTdense.py) and related attention modules, PASTIS transforms and loader, loss and metrics. `TSViT_DINOv3` subclasses TSViT and reuses `TSViT.forward()` with a new patch embedding.

### Commits (this fork)

| Commit | Summary |
| --- | --- |
| `af680b7` | DINOv3 patch embedding; both TSViT Transformers retained |
| `532341c` | Temporal-only variant without TSViT Spatial Transformer |
| `9bff7e6` | Documentation and smoke validation artifacts |

---

## Architectures

| `MODEL.architecture` | Pipeline |
| --- | --- |
| `TSViT` | Patch embedding → Temporal → Spatial → head (original) |
| `TSViT_DINOv3` | DINOv3 → Projection → Temporal → Spatial → head |
| `TSViT_DINOv3_Temporal` | DINOv3 → Projection → Temporal → head |

Both DINOv3 variants use RGB input through an adapter, keep the backbone frozen in `eval()` mode with `torch.no_grad()`, and train the projection, temporal blocks (and spatial where applicable), and head. Loss remains the original masked cross-entropy.

For RGB denormalization, resize to DINOv3 input size, padding-frame handling, and token grid alignment (12×12), see [README_DINOV3.md — Architecture](README_DINOV3.md#1-architecture).

---

## Installation

Tested on Linux with Python 3.13, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, and an NVIDIA GPU (CUDA 12.8). Run from the repository root:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-dinov3.txt
```

The legacy conda file `deepsatmodels_env.yml` comes from upstream; use the environment above for DINOv3 work.

Pin versions for reproduction: [`docs/validation/environment-freeze.txt`](docs/validation/environment-freeze.txt).

---

## Configuration and data

Prepare:

1. A local Hugging Face DINOv3 checkpoint directory (`config.json`, `model.safetensors`, `preprocessor_config.json`).
2. PASTIS24 pickle data and path CSVs (see [README_DINOV3.md §3–§4](README_DINOV3.md#3-environment-and-checkpoints)).

Minimal YAML fields:

| Field | Role |
| --- | --- |
| `MODEL.architecture` | `TSViT_DINOv3` or `TSViT_DINOv3_Temporal` |
| `MODEL.dinov3_checkpoint` | Path to checkpoint directory |
| `MODEL.frame_chunk_size` | Default `1` reduces VRAM (one frame through DINOv3 at a time) |
| `DATASETS.train/eval.base_dir`, `paths` | Data root and split CSVs |
| `CHECKPOINT.save_path` | Run output directory |

Update paths in `*_smoke.yaml` on your machine. `*_fold1.yaml` use `/path/to/...` placeholders for full training.

---

## Smoke test and training

After paths are set:

```bash
.venv/bin/python tests/smoke_dinov3.py \
  --config configs/PASTIS24/TSViT_DINOv3_Temporal_small_smoke.yaml

.venv/bin/python train_and_eval/segmentation_training_transf.py \
  --config configs/PASTIS24/TSViT_DINOv3_Temporal_small_smoke.yaml --device 0
```

Use `configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml` for the variant that keeps the TSViT Spatial Transformer.

Full fold training example (configure paths first):

```bash
.venv/bin/python train_and_eval/segmentation_training_transf.py \
  --config configs/PASTIS24/TSViT_DINOv3_Temporal_small_fold1.yaml --device 0
```

The training script expects CUDA. Original TSViT checkpoints cannot be loaded directly into DINOv3 architectures; use matching architecture YAML and weights.

CLI for this repo: `--config` and `--device` (not `--config_file` / `--gpu_ids` from [README_TSVIT.md](README_TSVIT.md)).

More commands, troubleshooting, and checkpoint layout: [README_DINOV3.md](README_DINOV3.md).

---

## Validation status (local)

DINOv3 Small passed forward/backward checks, one optimizer step, entrypoint train/eval, and checkpoint save/load on real sample `10000_0.pickle`.

| Architecture | Smoke |
| --- | --- |
| `TSViT_DINOv3` | Passed |
| `TSViT_DINOv3_Temporal` | Passed |

Setup: batch size 1, T=60, RGB, frozen backbone. These checks verify execution only—not segmentation quality. Reports:

- [Two-Transformer variant](docs/dinov3-smoke.md)
- [Temporal variant](docs/dinov3-temporal-smoke.md)

---

## References

- [DINOv3 integration guide](README_DINOV3.md)
- [Original TSViT documentation and PASTIS24 baselines](README_TSVIT.md)
- [Context-self contrastive learning](README_CSCL.md)
- [ViTs for SITS — Tarasiou et al., CVPR 2023](https://arxiv.org/abs/2301.04944)
- [DINOv3 — Meta](https://github.com/facebookresearch/dinov3)

DeepSatModels: [Apache License 2.0](LICENSE.txt). DINOv3 weights follow each checkpoint’s license.

Copyright © 2023 Michail Tarasiou.
