# TSViT with DINOv3 for Satellite Time-Series Segmentation

This fork integrates a **pretrained, frozen DINOv3 backbone** into [DeepSatModels](https://github.com/michaeltrs/DeepSatModels) (`53e6559`) for PASTIS24 segmentation. The changes focus on image feature extraction while retaining TSViT's temporal modeling and segmentation objective.

## What changed

Implementation: [`models/TSViT/TSViTDINOv3.py`](models/TSViT/TSViTDINOv3.py).

| Component | Original TSViT | `TSViT_DINOv3` | `TSViT_DINOv3_Temporal` |
| --- | --- | --- | --- |
| Patch embedding | Linear projection of 10-band patches | RGB adapter → frozen DINOv3 → trainable projection | Same DINOv3 integration |
| Temporal Transformer, DOY embedding and class tokens | Original | Retained | Retained |
| Spatial Transformer and spatial positional embedding | Original | Retained | Removed |
| Segmentation head | LayerNorm → Linear → patch reconstruction | Retained | Retained |
| Logits / loss | `[B,19,24,24]` / masked cross-entropy | Unchanged | Unchanged |

**`TSViT_DINOv3` replaces only `to_patch_embedding`**, inheriting the original `TSViT.forward()`. The temporal variant additionally removes the post-temporal spatial modules: DINOv3 supplies spatial context per date, before temporal aggregation.

Input is `[B,T,24,24,11]`: ten spectral bands and one DOY channel. The fixed RGB adapter selects B04/B03/B02 (indices `[2,1,0]`), reverses PASTIS normalization, divides reflectance by 10000, clamps, resizes to 192×192 and applies DINOv3 normalization. DINOv3 Small provides 144 patch tokens of dimension 384; a trainable `Linear(384,128)` maps them to TSViT's embedding space. CLS/register tokens are discarded. The other seven spectral bands are unused for image features.

```text
Original TSViT:
  Spectral patch embedding → Temporal Transformer → Spatial Transformer → Head

TSViT_DINOv3:
  RGB adapter → Frozen DINOv3 → Projection → Temporal Transformer → Spatial Transformer → Head

TSViT_DINOv3_Temporal:
  RGB adapter → Frozen DINOv3 → Projection → Temporal Transformer → Head
```

In all paths, DOY embeddings are added and learned class tokens prepended **before** the Temporal Transformer. Its class-token outputs feed the next stage. The retained spatial path adds spatial positions before dropout and spatial attention; the temporal variant applies dropout directly before the head. The shared head predicts 2×2 pixel logits per class/patch, reconstructed into `[B,19,24,24]`.

## Pretrained checkpoint

Source: Meta's [DINOv3 ViT-S/16, LVD-1689M](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m). The tested revision and file SHA256 are recorded in [validation provenance](docs/validation/final-task1/provenance.json). This is the web-pretrained Small model; the satellite-pretrained variant has not been tested.

After obtaining access on Hugging Face, place `config.json`, `model.safetensors` and `preprocessor_config.json` in one local directory. Set this field in both smoke YAMLs:

```yaml
MODEL:
  dinov3_checkpoint: /absolute/path/to/dinov3-small
```

`AutoModel.from_pretrained(..., local_files_only=True)` loads that directory without downloading weights. The backbone has `requires_grad=False`, runs inside `torch.no_grad()`, and stays in `eval()` after `net.train()`. The projection runs **outside** `no_grad()`. See the [environment and data setup](README_DINOV3.md#3-environment-and-checkpoints) before running tests.

## Loss and parameter updates

The **original masked cross-entropy objective is retained** because the segmentation classes, labels and output format are unchanged. Source: [`MaskedCrossEntropyLoss`](metrics/loss_functions.py) and the [`training loop`](train_and_eval/segmentation_training_transf.py).

For logits `z`, labels `y`, and valid pixels `V = {i : y_i != 19}`:

$$
\mathcal{L}=-\frac{1}{\lvert V\rvert}\sum_{i\in V}\log\left(\frac{\exp(z_{i,y_i})}{\sum_{c=0}^{18}\exp(z_{i,c})}\right),\qquad \lvert V\rvert>0.
$$

The training loop permutes logits to `[B,H,W,C]`; labels and boolean masks are `[B,H,W,1]`. The criterion selects valid logits/labels, applies log-softmax on the class axis and gathers target-class values, then averages over valid pixels. **Void label 19 is excluded; labels 0–18 participate.** No DINOv3 pretraining loss is added.

**All-void fix:** mean reduction now raises `ValueError` before backward/step instead of returning `NaN`. Valid-pixel values/gradients are unchanged; unreduced evaluation output remains empty when no pixels are valid. Returning zero alone would still allow AdamW momentum/weight-decay updates, so this fix explicitly rejects the batch.

```python
optimizer.zero_grad()
logits = net(inputs)
loss = criterion(logits.permute(0, 2, 3, 1), (labels, valid_mask))  # calculate loss
loss.backward()    # compute gradients; weights are unchanged
optimizer.step()   # AdamW updates weights using those gradients
```

The following results apply to the measured valid-pixel smoke step:

| Module | Trainable | Gradient | Weight Updated |
| --- | --- | --- | --- |
| RGB adapter | No learned parameters | Inside `no_grad()` | N/A |
| DINOv3 backbone | No | None | No |
| Projection | Yes | Finite, nonzero | Yes |
| DOY embedding and temporal class tokens | Yes | Finite, nonzero | Yes |
| Temporal Transformer | Yes | Finite, nonzero | Yes |
| Spatial Transformer and spatial positions | Yes, when retained | Finite, nonzero | Yes, when retained |
| Segmentation head | Yes | Finite, nonzero | Yes |

Fixed DINOv3 features still allow gradients with respect to the projection's weights and all subsequent trainable layers. On the tested single GPU, AdamW receives exactly the trainable parameters through `get_net_trainable_params`; the configs use `lr=1e-3` and `weight_decay=0`.

## Verification

Both variants passed real-data checks on 2026-10-18: DINOv3 Small, RTX 3050 Laptop, seed 42, one PASTIS24 sample, B=1, T=60, 521 valid pixels and 55 void pixels.

| Check | `TSViT_DINOv3` | `TSViT_DINOv3_Temporal` |
| --- | --- | --- |
| Logits shape | `[1,19,24,24]` | `[1,19,24,24]` |
| Finite loss before update | 3.0117473602 | 2.8545644283 |
| Optimizer tensors / trainable tensors (no duplicates) | 102/102 | 55/55 |
| Trainable tensors changed after step | 102/102 | 55/55 |
| Backbone tensors match local pretrained checkpoint | 211/211 | 211/211 |
| Backbone state after step | Unchanged | Unchanged |
| Forward, backward, AdamW step and checkpoint round trip | Passed | Passed |

Snapshots confirmed that weights stayed unchanged after `backward()` and changed after `step()`. Selected group norms are below; each cell is **gradient L2 / parameter-change L2**, accumulated across all elements in the group in float64. The JSON's legacy `gradient_norms` field separately retains the sum of per-tensor norms.

| Module | `TSViT_DINOv3` | `TSViT_DINOv3_Temporal` |
| --- | ---: | ---: |
| Projection | 0.205408 / 0.221944 | 0.304522 / 0.221953 |
| Temporal Transformer | 1.238322 / 0.889630 | 1.844437 / 0.889687 |
| Spatial Transformer | 1.276541 / 0.889424 | Removed |
| Segmentation head | 1.888751 / 0.025311 | 2.310214 / 0.025314 |

Reproducible results from the updated smoke test: [DINOv3 JSON](docs/validation/final-task1/dinov3.json), [temporal JSON](docs/validation/final-task1/temporal.json). Four CPU regression tests also pass, covering valid-pixel values/gradients and all-void handling. Earlier reports remain intact: [two-Transformer](docs/dinov3-smoke.md), [temporal](docs/dinov3-temporal-smoke.md). These checks establish execution and parameter updates; **they do not establish improved accuracy**.

## Files updated from upstream

| Files | Change |
| --- | --- |
| [`models/TSViT/TSViTDINOv3.py`](models/TSViT/TSViTDINOv3.py) | RGB adapter, frozen DINOv3, projection and both variants |
| [`models/__init__.py`](models/__init__.py) | Register both architectures; defer optional FCN import |
| [`configs/PASTIS24/TSViT_DINOv3*.yaml`](configs/PASTIS24/) | Smoke and fold1 configs |
| [`tests/smoke_dinov3.py`](tests/smoke_dinov3.py), [`tests/fixtures/`](tests/fixtures/) | Real-sample integration checks |
| [`metrics/loss_functions.py`](metrics/loss_functions.py), [`tests/test_masked_cross_entropy.py`](tests/test_masked_cross_entropy.py) | Reject empty mean loss; explicit class axis; preserve valid-pixel behavior with regression tests |
| [`train_and_eval/segmentation_training_transf.py`](train_and_eval/segmentation_training_transf.py) | Seed, optional CUDA memory reporting and writer cleanup |
| [`data/__init__.py`](data/__init__.py), [`data/datasets.yaml`](data/datasets.yaml), [`utils/config_files_utils.py`](utils/config_files_utils.py) | Configurable paths, deferred France loader import, YAML compatibility fixes |
| [`requirements-dinov3.txt`](requirements-dinov3.txt), [`README_DINOV3.md`](README_DINOV3.md), [`docs/`](docs/) | Dependencies, integration details and validation reports |

The original TSViT blocks and PASTIS transforms/loader are unchanged. The loss formula is preserved; the empty-mean guard and explicit log-softmax axis are the only loss-code changes.

<a id="installation"></a>

## Run the smoke tests

Follow the [environment, checkpoint and data setup](README_DINOV3.md#3-environment-and-checkpoints). Set the local checkpoint/data paths in the smoke YAMLs, then run from the repository root with CUDA:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_masked_cross_entropy.py' -v
.venv/bin/python tests/smoke_dinov3.py \
  --config configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml \
  --output outputs/task1-final/dinov3
.venv/bin/python tests/smoke_dinov3.py \
  --config configs/PASTIS24/TSViT_DINOv3_Temporal_small_smoke.yaml \
  --output outputs/task1-final/temporal
```

The CPU tests need no data/checkpoint. Each GPU smoke writes `result.json` with loss, norms, deltas, optimizer coverage, checkpoint/source hashes and frozen-state checks, plus a round-trip checkpoint. Outputs are gitignored; checkpoint downloads and full training are not part of these commands.

## Current limitations

- Full training, held-out accuracy/mIoU, multi-GPU execution and controlled comparisons with original TSViT remain untested. Only DINOv3 Small has been verified.
- RGB selection discards seven spectral bands. The original temporal padding policy is retained without an attention mask.
- An all-void training batch now stops with an explicit error; automatic skipping is not implemented. An evaluation split with no valid labels still has undefined aggregate metrics.

Original documentation: [TSViT](README_TSVIT.md). Code license: [Apache 2.0](LICENSE.txt); pretrained weights follow their own license. Copyright © 2023 Michail Tarasiou.
