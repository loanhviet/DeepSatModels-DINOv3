# Kiểm chứng DINOv3 + Temporal TSViT — 2026-10-18

Bản nền thay patch embedding đã được lưu ở commit `af680b7`. Biến thể mới
`TSViT_DINOv3_Temporal` giữ DINOv3 Small, projection, temporal Transformer,
DOY, class tokens và head; bỏ spatial Transformer và spatial positional
embedding của TSViT. Không thay loss, data pipeline hoặc train/eval loop.

```text
RGB từng ngày → DINOv3 đóng băng → Projection + embedding DOY
  → Temporal Transformer TSViT → head + ghép patch → bản đồ 19 lớp
```

Đây là thay encoder không gian, vẫn giữ encoder thời gian TSViT. DINOv3 xử
lý không gian trước khi gộp thời gian; bản này không còn spatial encoder
hậu temporal. [README](../README_DINOV3.md) mô tả đầy đủ hai architecture.

## Kiểm tra đã chạy

Mẫu thật `10000_0.pickle` từ cùng thư mục Drive người dùng cung cấp: 43
quan sát + 17 frame pad, 521 pixel hợp lệ. Batch 1, T=60; seed 42. GPU RTX
3050 Laptop 4096 MiB, Python 3.13.14, torch 2.9.1+cu128, Transformers 4.57.6;
dùng lại môi trường của bản đầu.

| Kiểm tra | Kết quả |
| --- | --- |
| Input / logits | `[1,60,24,24,11]` / `[1,19,24,24]` |
| Forward, masked cross-entropy, backward, AdamW | Qua; loss `2.8545644283` |
| Gradient | Projection, DOY embedding, temporal Transformer, class tokens và head hữu hạn, khác 0; trọng số được cập nhật |
| DINOv3 frozen | Không có gradient, hash trọng số không đổi; vẫn eval sau net.train() |
| Spatial encoder | Không còn module và không có key `space_*` trong state_dict |
| Padding | Giữ zero feature trước projection, bias sau projection; không thêm temporal attention mask |
| RGB / token order | Qua đối chiếu raw RGB, B/T/patch và chunk 1/2 |
| Kiểm tra forward độc lập | Khớp chính xác với TSViT.forward gốc khi spatial = Identity, spatial position = 0 |
| Batch lớn hơn 1 | Kiểm tra độc lập trên 2 chuỗi, mỗi chuỗi 2 ngày thật khác nhau: khớp chính xác |
| Save/load | Qua loader gốc; logits nạp lại bằng chính xác |
| Entrypoint gốc | Exit 0; 1 train step + 1 eval pass, lưu checkpoint và TensorBoard |
| Train / eval loss | `2.8545644283` / `2.1593797` |
| Checkpoint entrypoint / bước AdamW độc lập | Cả 271 state tensors bằng chính xác |
| Regression bản giữ hai Transformer | Qua toàn bộ smoke; 318 state tensors sau AdamW khớp checkpoint kiểm tra ở commit đầu |

Giữ nguyên byte nội dung các file TSViT encoder/attention, data PASTIS và
loss/metrics gốc; đối chiếu checksum với báo cáo trước. Head và bộ tổng hợp
thời gian dùng cấu trúc gốc. Forward của biến thể mới được viết riêng để
bỏ hai phép xử lý spatial, không sửa `TSViT.forward()` gốc.

## Tham số và VRAM

| Số lượng | Bản giữ cả hai Transformer | Biến thể temporal |
| --- | ---: | ---: |
| Tham số được train | 1.701.508 | 891.268 |
| Tham số backbone đóng băng | 21.596.544 | 21.596.544 |
| Peak allocated, integration | 687,54 MiB | 574,32 MiB |
| Peak reserved, integration | 754 MiB | 656 MiB |
| Peak allocated, entrypoint | 670,93 MiB | 569,58 MiB |
| Peak reserved, entrypoint | 748 MiB | 648 MiB |

VRAM đo bằng peak allocator PyTorch; không bao gồm CUDA context/cấp phát
ngoài PyTorch. Integration có các bản sao trọng số phục vụ assertion; vùng
đo không bao gồm model tham chiếu hay round-trip checkpoint. Số entrypoint
của bản giữ hai Transformer lấy từ lần kiểm chứng trước, cùng GPU/môi trường.

**Train/eval dùng cùng một mẫu để kiểm tra pipeline. Chưa đánh giá chất
lượng.** Không so sánh kiến trúc dựa trên các loss này: bộ spatial bị bỏ làm
thay đổi cả chuỗi khởi tạo ngẫu nhiên của head. Cần train/validation tách
biệt và cùng điều kiện để đánh giá việc bỏ spatial có lợi hay không.

## Lệnh chạy và artifact

```bash
.venv/bin/python tests/smoke_dinov3.py \
  --config configs/PASTIS24/TSViT_DINOv3_Temporal_small_smoke.yaml
.venv/bin/python train_and_eval/segmentation_training_transf.py \
  --config configs/PASTIS24/TSViT_DINOv3_Temporal_small_smoke.yaml --device 0
.venv/bin/python tests/smoke_dinov3.py \
  --config configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml \
  --output outputs/dinov3-backbone-regression
```

Đường dẫn mẫu, checkpoint DINOv3 và output train nằm trong YAML. Full-fold
config riêng: `configs/PASTIS24/TSViT_DINOv3_Temporal_small_fold1.yaml`, cần
điền đường dẫn full dataset/split thật trước khi chạy. Checkpoint hai biến
thể không strict-load chéo; dùng thư mục output riêng.

- [Integration](validation/temporal/integration.json),
  [entrypoint VRAM](validation/temporal/entrypoint-cuda.json),
  [log entrypoint](validation/temporal/entrypoint.log).
- [Regression bản đầu](validation/temporal/backbone-regression.json),
  [đối chiếu checkpoint](validation/temporal/checkpoint-comparison.json),
  [provenance](validation/temporal/provenance.json).
- Artifact runtime được gitignore trong `outputs/dinov3-temporal-integration/`,
  `outputs/dinov3-temporal-smoke-entrypoint/` và `outputs/dinov3-backbone-regression/`.

Không thêm dependency, không sửa `crop-map`, không tải lại checkpoint/dataset,
không push GitHub. `git diff --check`, compile các file Python thay đổi và
parse các config YAML đều qua.
