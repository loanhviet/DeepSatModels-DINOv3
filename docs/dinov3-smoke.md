# Kiểm chứng DINOv3 Small — 2026-10-17

Thực hiện trên repo gốc tại `53e6559`, với thay đổi patch embedding được mô tả
trong [README_DINOV3.md](../README_DINOV3.md). Không chạy full training.

## Dữ liệu và môi trường

- Mẫu thật `10000_0.pickle` từ [Drive PASTIS24](https://drive.google.com/drive/u/0/folders/1-ymz_s6-3BCUcLqNDJ67VKiSTx2n4Glw),
  file ID `1yArZczqDkt8K9FrD1I8Jwsu5Ww72XveO`, 497.798 byte.
- SHA256 mẫu: `4852e58f320e8b1d7aedba527fa427870d9eeb434892a1931e549150931f3766`.
- 43 quan sát thật + 17 frame padding; 521/576 pixel có nhãn hợp lệ,
  55 pixel void (19). Batch 1, T=60; RGB B04/B03/B02, backbone đóng băng.
- Python 3.13.14, torch 2.9.1+cu128, torchvision 0.24.1+cu128,
  Transformers 4.57.6; RTX 3050 Laptop 4096 MiB; driver 595.99.02.
- [Danh sách phiên bản đầy đủ](validation/environment-freeze.txt),
  [provenance/checksum](validation/provenance.json).

## Kết quả

| Kiểm tra | Kết quả |
| --- | --- |
| Đầu vào / logits | `[1,60,24,24,11]` / `[1,19,24,24]` |
| RGB adapter so với RGB từ raw pixels | Sai số lớn nhất 0 trên frame kiểm tra |
| Forward + loss + backward + AdamW | Qua; loss ban đầu `3.0117473602` |
| Gradient | Projection, temporal embedding, hai Transformer, head đều hữu hạn và khác 0 |
| Cập nhật trọng số | Các nhóm trainable được cập nhật; backbone không đổi hash, không có gradient |
| Padding | 17 frame pad và đầu vào toàn pad đều cho embedding sau projection bằng bias |
| Attention | Kế thừa đúng `TSViT.forward`; temporal/spatial dùng `Transformer` và `Attention` gốc |
| Thứ tự token | Qua kiểm tra B/T/patch và chunk 1 so với chunk 2 |
| Save/load model mới | Qua loader checkpoint gốc, logits nạp lại bằng chính xác |
| TSViT gốc | Forward/backward qua cùng mẫu và loss; logits đúng shape, loss `3.0058653355` |
| Entrypoint gốc | Exit 0; 1 train step + 1 eval pass + checkpoint/TensorBoard |
| Train / eval loss qua entrypoint | `3.0117473602` / `2.2503839` |
| Checkpoint entrypoint so với bước AdamW độc lập | Cả 318 state tensors bằng chính xác |

Seed 42. **Train và eval dùng cùng một mẫu**: loss/metrics này chỉ xác nhận
pipeline chạy được, không dùng để kết luận chất lượng hoặc so sánh model.
Chưa chạy Base/Satellite, full dataset hoặc đánh giá split chính thức.

### VRAM đo được

| Lần kiểm tra | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| Integration forward/backward và các assertion | 687,54 MiB | 754 MiB |
| Entrypoint train + eval | 670,93 MiB | 748 MiB |

Đo bằng `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`,
1 MiB = 2²⁰ byte. Đây là VRAM của allocator PyTorch, không phải toàn bộ
bộ nhớ tiến trình theo `nvidia-smi`; không bao gồm CUDA context, các cấp phát
ngoài PyTorch hoặc desktop. Integration có các bản sao trainable weights
để kiểm tra cập nhật nên peak hơi cao hơn entrypoint.

Thời gian vùng đo: integration 1,73 giây, entrypoint 0,95 giây. Integration
bao gồm nạp model và các assertion trước/sau bước train, sau khi kiểm tra
TSViT gốc đã làm nóng GPU; không bao gồm round-trip checkpoint. Entrypoint
đo vùng train/eval sau khi đã nạp model. Không coi đây là benchmark tốc độ.

## Tái lập và artifact

Chạy từ root repo, với đường dẫn dữ liệu/checkpoint trong
`configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml`:

```bash
.venv/bin/python tests/smoke_dinov3.py \
  --config configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml
.venv/bin/python train_and_eval/segmentation_training_transf.py \
  --config configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml --device 0
```

- [Integration JSON](validation/integration.json),
  [CUDA entrypoint JSON](validation/entrypoint-cuda.json),
  [log entrypoint](validation/entrypoint.log),
  [đối chiếu checkpoint](validation/entrypoint-comparison.json).
- Artifact runtime trong `outputs/dinov3-integration/` và
  `outputs/dinov3-smoke-entrypoint/`: có `roundtrip.pth`, `1epoch_1step.pth`,
  `best.pth`, config chạy, log và TensorBoard. `outputs/` được gitignore.
- `git diff --check` qua; cả 25 experiment YAML parse được bằng `safe_load`.
  Các file TSViT encoder/attention, data PASTIS, loss/metrics gốc được đối
  chiếu byte-for-byte với commit gốc, checksum nằm trong provenance JSON.

Dữ liệu/checkpoint nguồn trong `crop-map` chỉ được đọc, không chỉnh sửa.
Không tải lại checkpoint, không tải toàn bộ dataset, không commit/push GitHub.
