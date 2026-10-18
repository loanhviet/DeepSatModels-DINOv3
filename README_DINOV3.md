# Task 1: DINOv3 patch embedding cho TSViT

Bản gốc: commit `53e655936be088522ec419b3ead38684c20ae1ec`.
Mục tiêu đầu tiên là chạy được semantic segmentation PASTIS24.
**Thay patch embedding bằng backbone DINOv3 trích đặc trưng từng ngày;
không thay toàn bộ encoder TSViT.** Bản này chỉ dùng RGB và đóng băng DINOv3.

## Kiến trúc và các thay đổi

```text
Input gốc [B,T,24,24,11] = 10 band chuẩn hóa PASTIS + DOY
  ├─ DOY → temporal embedding gốc (366 → 128)
  └─ 10 band → RGB adapter → resize 192×192 → DINOv3 /16 đóng băng
       → bỏ CLS/register tokens → [B,T,144,D] → Linear(D,128)
       → [(B×144),T,128]
  → cộng temporal embedding
  → temporal Transformer + 19 token theo lớp, giữ nguyên
  → spatial position embedding + spatial Transformer, giữ nguyên
  → mlp_head (128 → 4) + dựng lại patch 2×2, giữ nguyên
  → logits [B,19,24,24] → masked cross-entropy gốc
```

`models/TSViT/TSViTDINOv3.py` thêm subclass `TSViTDINOv3(TSViT)` và chỉ
thay `self.to_patch_embedding`. Subclass **kế thừa nguyên `TSViT.forward()`**;
không sửa `TSViTdense.py`, `module.py`, transforms PASTIS hay loss/metrics.
Cả temporal và spatial attention vẫn là `Attention` gốc: Q/K/V, softmax,
residual và LayerNorm như trước; không thêm attention mask.

Adapter chọn B04/B03/B02 ở chỉ số `[2,1,0]` theo
[notebook PASTIS chính thức](https://github.com/VSainteuf/pastis-benchmark/blob/main/demo.ipynb).
Vì dataset đã chuẩn hóa, adapter tái sử dụng đúng mean/std từ `Normalize`
gốc để tính `raw = normalized * std + mean`. Sau đó tính
`rgb = clamp(raw / 10000, 0, 1)`, resize bilinear lên 192×192 rồi chuẩn hóa
bằng `image_mean`/`image_std` trong `preprocessor_config.json` của checkpoint.
Không chia thêm cho 255, không min-max từng ảnh. Chuyển thang phản xạ này là
lựa chọn cho bản chạy đầu, chưa tối ưu chất lượng. Bảy band còn lại không
tham gia trích đặc trưng; DOY vẫn dùng riêng như TSViT gốc.

192/16 = 12 patch mỗi chiều, khớp lưới gốc 24/2 = 12; resize chỉ diễn ra
trong adapter, không resize nhãn. Lấy token sau `1 + num_register_tokens`
theo [giao diện DINOv3 Transformers](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/dinov3).
Projection dùng D từ config checkpoint: Small 384, Base 768, Satellite 1024.
Small/Base dùng mean/std ImageNet; Satellite dùng mean/std riêng, được đọc
tự động từ file preprocessing. Chỉ Small đã được chạy kiểm chứng.

### Padding sau projection và attention

Transform gốc chuẩn hóa trước rồi pad toàn bộ frame, kể cả DOY, bằng 0.
Adapter nhận diện frame có cả 10 band đã chuẩn hóa bằng 0, bỏ qua DINOv3 và
điền **feature bằng 0 trước projection**. Vì projection có bias:

```text
patch embedding của frame padding = Linear(0) = projection.bias
đầu vào temporal = projection.bias + temporal_embedding(one_hot(DOY=0))
```

**Không zero lại embedding sau projection.** Cách này tương ứng với Linear
nhúng patch gốc, vốn cũng trả về bias khi patch bằng 0. `DOY=0` vẫn là một
one-hot được nhúng, không phải vector temporal embedding bằng 0. Các vị trí
padding vẫn tham gia attention như TSViT gốc; `seq_lengths` không được thêm
vào model và không dùng để tạo mask. Kiểm tra cả 17 frame padding của mẫu
thật và trường hợp đầu vào toàn padding đã qua.

Backbone có `requires_grad=False`, luôn ở `eval()` kể cả sau `net.train()`;
chỉ trích đặc trưng bên trong `torch.no_grad()`. Projection đặt bên ngoài
vùng này. Các temporal/spatial Transformer, positional embeddings, token và
head vẫn được train. `frame_chunk_size=1` giới hạn số ảnh qua DINO mỗi lượt.

## Môi trường riêng

Chạy từ thư mục gốc repo. Không dùng hoặc thay môi trường `crop-map/.venv`.
Bộ conda Python 3.8/CUDA 10.1 gốc không phù hợp với nhánh DINOv3 này.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-dinov3.txt
```

Cặp torch/torchvision lấy từ [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/).
Đã kiểm chứng Python 3.13.14, torch 2.9.1+cu128, torchvision 0.24.1+cu128,
Transformers 4.57.6, RTX 3050 Laptop 4096 MiB. Phiên bản đầy đủ được lưu
trong [báo cáo môi trường](docs/validation/environment-freeze.txt).

Có thể thêm `--find-links /home/viet/projects/TLU/TSViT/crop-map/preparation/dinov3/wheelhouse`
vào lệnh cài dependency để tái sử dụng các wheel đã có. Đây không phải bộ
wheel CUDA đầy đủ. Checkpoint được đọc tại chỗ qua
`AutoModel.from_pretrained(..., local_files_only=True)`; không tải lại và
không dùng `torch.hub` với file safetensors.

## Config và cách chạy

Mọi đường dẫn dữ liệu/checkpoint nằm trong YAML, không hardcode trong model
hoặc script kiểm chứng. DINOv3 yêu cầu `MODEL.dinov3_checkpoint` trỏ tới thư
mục Hugging Face gồm `config.json`, `model.safetensors`,
`preprocessor_config.json`. Có thể đổi đường dẫn sang Base/Satellite sau;
backbone vẫn đóng băng. Thiếu file sẽ báo lỗi, không tự tải mạng.

Config đầy đủ: `configs/PASTIS24/TSViT_DINOv3_small_fold1.yaml`.
Sửa `DATASETS.train/eval.base_dir` và `paths` thành vị trí dữ liệu/split
thật. `paths` là CSV một cột, không header; đường dẫn từng pickle được ghép
với `base_dir` như dataloader gốc. Khi không cung cấp hai trường này, loader
tiếp tục dùng `data/datasets.yaml` như trước. Config mẫu fold1 đầy đủ còn để
`/path/to/PASTIS24` vì chưa tải toàn bộ tập và split chính thức vào repo.

```bash
.venv/bin/python train_and_eval/segmentation_training_transf.py \
  --config configs/PASTIS24/TSViT_DINOv3_small_fold1.yaml --device 0
```

Batch train/eval mặc định 1, `num_workers=0`, T=60; temporal/spatial depth=4,
dim=128, 19 lớp như config TSViT gốc. Để trống
`CHECKPOINT.load_from_checkpoint` trong lần đầu. Checkpoint TSViT cũ không
strict-load trực tiếp được vào model mới do patch embedding khác nhau.
Checkpoint mới vẫn lưu đầy đủ `state_dict` bằng pipeline gốc; nạp lại trọng
số chưa phải resume trạng thái optimizer/scheduler.

### Smoke check trên mẫu thật

`configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml` trỏ tới mẫu đã có
`10000_0.pickle`, nguồn [Drive PASTIS24 được cung cấp](https://drive.google.com/drive/u/0/folders/1-ymz_s6-3BCUcLqNDJ67VKiSTx2n4Glw).
Chỉ đọc dữ liệu và checkpoint từ `crop-map`; CSV, log, TensorBoard và
checkpoint thử đều nằm trong repo hiện tại. Không cần tải lại mẫu.

```bash
# RGB, padding, gradient, backbone frozen, token ordering, checkpoint round-trip
.venv/bin/python tests/smoke_dinov3.py \
  --config configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml

# Entrypoint gốc: 1 epoch/1 train step, eval và save mỗi step
.venv/bin/python train_and_eval/segmentation_training_transf.py \
  --config configs/PASTIS24/TSViT_DINOv3_small_smoke.yaml --device 0
```

`CHECKPOINT.profile_cuda: true` ghi peak allocated/reserved và train loss
vào `CHECKPOINT.save_path/cuda_memory.json`; mặc định không bật cho config
cũ. Mức này đo bằng bộ cấp phát PyTorch, không bao gồm toàn bộ CUDA context,
thư viện CUDA hoặc tiến trình hiển thị. Không dùng làm bảo đảm VRAM cho batch
lớn hơn hoặc backbone khác.

Kết quả và lệnh tái lập được ghi tại [docs/dinov3-smoke.md](docs/dinov3-smoke.md).
Train và eval dùng **cùng một mẫu** trong smoke check, chỉ xác nhận pipeline
hoạt động, không phải đánh giá chất lượng hay kết quả 5-fold PASTIS24.

## Các sửa tương thích cần thiết

- Model factory thêm architecture mới; import `FCN_CRNN` chỉ khi dùng nhánh
  tương ứng để TSViT không cần nạp `torchfcn` cũ.
- Data factory cho phép override đường dẫn trong config và chỉ import
  dataloader France khi chọn France: file France gốc tại commit này có lỗi
  indentation, trước đó làm cả import PASTIS thất bại. Không sửa nhánh France.
- `read_yaml` dùng `safe_load` cho PyYAML mới; sửa cấp lồng thừa của fold1
  trong `data/datasets.yaml`.
- Entrypoint segmentation thêm seed tùy chọn, đóng SummaryWriter khi xong
  và xuất VRAM khi bật profiling. Train/eval/loss/scheduler vẫn là pipeline
  gốc. Smoke một bước có cảnh báo cosine scheduler không tạo lịch thay đổi
  đáng kể, điều này phù hợp với lần kiểm tra rất ngắn.
- Cờ CLI đúng ở commit này là `--config`, `--device` (ví dụ phía trên);
  README TSViT cũ ghi `--config_file`, `--gpu_ids` không khớp script thực tế.

Không thay dữ liệu, checkpoint hay code trong `crop-map`; chưa đẩy GitHub.
