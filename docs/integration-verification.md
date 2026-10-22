# One-step DINOv3 integration verification

This record verifies executable integration, not trained prediction quality. Existing local DINOv3 weights and one PASTIS24 sample were reused. No full training or model/data download was performed.

The checks were run on **2026-10-22**, on branch `feat/dinov3-integration`, with application code at `3355fe1`. [Provenance](validation/readme-review/provenance.json) records full commit identity, source hashes, commands and the tested scope.

## Environment and fixture

- Ubuntu 24.04.5 LTS; Python 3.13.14; PyTorch 2.9.1+cu128; Transformers 4.57.6.
- NVIDIA GeForce RTX 3050 Laptop GPU; existing frozen DINOv3 ViT-S/16 checkpoint.
- Real fixture: `10000_0.pickle`, 43 acquisitions and 17 padded dates; batch size 1, T = 60.
- Labels: 521 valid pixels and 55 void pixels on a 24 × 24 grid; seed 42.

## Loss checks on CPU

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_masked_cross_entropy.py' -v
```

Observed: **4 tests passed**, exit code 0. Valid-pixel values/gradients match cross-entropy, void pixels contribute zero logit gradient, the no-mask case is supported, and an all-void mean stops before any optimizer update. These are existing tests; no new application tests or model changes were needed for this documentation review.

## Dedicated real-sample smoke checks

Commands run from the repository root using existing local paths in the checked-in smoke configs:

```bash
.venv/bin/python tests/smoke_dinov3.py \
  --config configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml \
  --output outputs/readme-review/dinov3
.venv/bin/python tests/smoke_dinov3.py \
  --config configs/PASTIS24/TSViT_DINOv3_Temporal_small_smoke.yaml \
  --output outputs/readme-review/temporal
```

For another machine, the [README setup](../README.md#setup) creates local config copies with its checkpoint and sample paths. The same model, data transforms, loss and smoke parameters are used.

| Observation | Temporal + spatial | Temporal only |
| --- | --- | --- |
| Report | [dinov3.json](validation/readme-review/dinov3.json) | [temporal.json](validation/readme-review/temporal.json) |
| Exit code | 0 | 0 |
| Input shape | `[1,60,24,24,11]` | `[1,60,24,24,11]` |
| Logits shape | `[1,19,24,24]` | `[1,19,24,24]` |
| Loss before the step | 3.0117473602 | 2.8545644283 |
| Trainable / optimizer tensor coverage | 102 / 102 | 55 / 55 |
| Trainable parameter values | 1,701,508 | 891,268 |
| Frozen DINOv3 parameter values | 21,596,544 | 21,596,544 |
| Loaded pretrained state tensors match | 211 / 211 | 211 / 211 |
| Weights unchanged after backward, before step | Yes | Yes |
| Trainable tensors changed after AdamW step | 102 / 102 | 55 / 55 |
| DINOv3 state unchanged after step | Yes | Yes |
| Loss / logit gradients match valid-pixel cross-entropy | Yes | Yes |
| Void-pixel gradient maximum | 0 | 0 |
| Exact checkpoint round trip | Yes | Yes |

The test also checks RGB adaptation against raw pixels, batch/time/patch order, frame chunking, padding and the original TSViT regression path. The temporal-only variant matches the original forward with its spatial path bypassed, including a two-sample layout check.

Loss values are execution diagnostics from randomly initialized trainable modules. Their difference is not evidence that either variant is more accurate. GPU timing/memory fields in each JSON describe that smoke path and depend on hardware and runtime state.

## Standard training entrypoint

The config-generation snippet in README section 5.4 was executed with only the two path placeholders replaced by existing local paths. Both generated configs were checked to have one epoch, batch size 1 and a single-row sample CSV before launching:

```bash
.venv/bin/python train_and_eval/segmentation_training_transf.py \
  --config outputs/local-smoke/TSViT_DINOv3_small_smoke.yaml --device 0
.venv/bin/python train_and_eval/segmentation_training_transf.py \
  --config outputs/local-smoke/TSViT_DINOv3_Temporal_small_smoke.yaml --device 0
```

Both exited with code 0 after one training batch and one same-sample evaluation batch. Each generated `config_file.yaml`, `1epoch_1step.pth`, `best.pth`, TensorBoard events and `cuda_memory.json` in its configured output directory.

| Record | Temporal + spatial | Temporal only |
| --- | --- | --- |
| Entrypoint profile | [JSON](validation/readme-review/dinov3-entrypoint.json) | [JSON](validation/readme-review/temporal-entrypoint.json) |
| Entrypoint log | [Log](validation/readme-review/dinov3-entrypoint.log) | [Log](validation/readme-review/temporal-entrypoint.log) |
| Training steps | 1 | 1 |
| Last training loss | 3.0117473602 | 2.8545644283 |

The profiles confirm one training step. Same-sample evaluation metrics in the raw logs only establish that the evaluation loop executes; they are not held-out results. A one-step cosine-schedule warning does not affect the verified forward/loss/update/save-load mechanics.

Large checkpoint files and local configs remain under ignored `outputs/`. The small JSON/log records above are copied here for review. Earlier [Task 1 reports](validation/final-task1/) are preserved.
