# 🚀 Hướng Dẫn Chạy Dự Án - Nhận Diện Khuôn Mặt & Mũ Bảo Hiểm

## 📋 Mục lục
1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Cài đặt từ đầu](#cài-đặt-từ-đầu)
3. [Chạy ứng dụng cơ bản](#chạy-ứng-dụng-cơ-bản)
4. [Chạy ứng dụng với mũ bảo hiểm](#chạy-ứng-dụng-với-mũ-bảo-hiểm)
5. [Training model mũ bảo hiểm](#training-model-mũ-bảo-hiểm)
6. [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)

---

## 🖥️ Yêu cầu hệ thống

### Phần mềm:
- **Python 3.8+**: [Tải Python](https://www.python.org/downloads/)
- **Git** (tùy chọn): [Tải Git](https://git-scm.com/downloads)

### Phần cứng:
- **RAM**: Tối thiểu 4GB, khuyến nghị 8GB+
- **GPU**: NVIDIA GPU với CUDA (tùy chọn, để tăng tốc độ)
- **Camera**: Webcam hoặc camera USB (cho chế độ camera trực tiếp)

---

## 📦 Cài đặt từ đầu

### Bước 1: Mở Terminal/Command Prompt

**Windows:**
- Nhấn `Win + R`, gõ `cmd` hoặc `powershell`, nhấn Enter
- Hoặc nhấn `Win + X`, chọn "Windows PowerShell" hoặc "Command Prompt"

**macOS/Linux:**
- Mở Terminal (`Ctrl + Alt + T` trên Linux, `Cmd + Space` rồi gõ "Terminal" trên Mac)

### Bước 2: Di chuyển đến thư mục dự án

```bash
cd C:\Python\DuAnHocMay\NHANDIENMATNGUOI_YOLO
```

Hoặc nếu bạn đã clone/download dự án ở vị trí khác:
```bash
cd <đường-dẫn-đến-dự-án>
```

### Bước 3: Tạo môi trường ảo (Khuyến nghị)

**Trên Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Trên macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**✅ Khi thành công, bạn sẽ thấy `(venv)` ở đầu dòng lệnh:**
```
(venv) PS C:\Python\DuAnHocMay\NHANDIENMATNGUOI_YOLO>
```

### Bước 4: Cài đặt các thư viện

#### Cài đặt thư viện cơ bản:
```bash
pip install -r requirements.txt
```

#### Cài đặt TensorFlow (Bắt buộc cho helmet detection):
```bash
pip install tensorflow==2.13.0
```

**Hoặc cài đặt tất cả một lúc:**
```bash
pip install -r requirements.txt
pip install tensorflow==2.13.0 scikit-learn scipy
```

### Bước 5: Tải model YOLO (Tự động)

Lần đầu chạy ứng dụng, YOLO sẽ tự động tải model `yolov8n.pt` (khoảng 6MB).

**Nếu muốn tải thủ công:**
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## 🎯 Chạy ứng dụng cơ bản

### Phương án 1: Sử dụng Batch File (Windows - Khuyến nghị)

**Chạy ứng dụng nhận diện đầu người:**
```bash
run_head_detection.bat
```

**Chạy ứng dụng nhận diện đầu + mũ bảo hiểm:**
```bash
run_helmet_detection.bat
```

Sau khi chạy batch file, sẽ có menu để chọn:
- **Option 1**: Chạy ứng dụng nhận diện + mũ bảo hiểm
- **Option 2**: Train model mũ bảo hiểm
- **Option 3**: Chạy ứng dụng cũ (không có mũ bảo hiểm)

### Phương án 2: Chạy trực tiếp bằng Python

**Chạy ứng dụng nhận diện đầu người:**
```bash
python apps/head_detection_app.py
```

**Chạy ứng dụng nhận diện đầu + mũ bảo hiểm (Khuyến nghị):**
```bash
python apps/head_detection_app_with_helmet.py
```

**Chạy ứng dụng refactored:**
```bash
python apps/head_detection_app_refactored.py
```

### Phương án 3: Test model YOLO trước

Trước khi chạy ứng dụng, bạn có thể test model YOLO:
```bash
python apps/demo_test.py
```

---

## 🪖 Chạy ứng dụng với mũ bảo hiểm

### Bước 1: Kiểm tra model mũ bảo hiểm

Model mũ bảo hiểm có thể ở các vị trí:
- `helmet_detection_project/models/helmet_detection_model.h5`
- `helmet_detection_project/models/best_helmet_model.h5`
- `models/helmet_detection_model.h5`

**Kiểm tra xem model có tồn tại không:**
```bash
# Windows
dir helmet_detection_project\models\*.h5

# Linux/Mac
ls helmet_detection_project/models/*.h5
```

### Bước 2A: Nếu model đã có sẵn

Chạy ứng dụng:
```bash
python apps/head_detection_app_with_helmet.py
```

Hoặc:
```bash
run_helmet_detection.bat
# Chọn option 1
```

### Bước 2B: Nếu model chưa có - Training model

Xem phần [Training model mũ bảo hiểm](#training-model-mũ-bảo-hiểm) bên dưới.

### Bước 3: Copy model vào thư mục chính (nếu cần)

Nếu model ở trong `helmet_detection_project/models/`, bạn có thể copy vào thư mục chính:
```bash
python apps/copy_helmet_model.py
```

---

## 🎓 Training model mũ bảo hiểm

### Bước 1: Thu thập dữ liệu

**Chạy ứng dụng thu thập dữ liệu:**
```bash
cd helmet_detection_project
python data_collection/data_collection_app.py
```

**Hoặc sử dụng ứng dụng chính để chụp ảnh:**
1. Chạy `python apps/head_detection_app_with_helmet.py`
2. Chọn "Bắt đầu Camera"
3. Ảnh sẽ được lưu vào `captured_heads/`
4. Phân loại ảnh vào `helmet_detection_project/data_collection/with_helmet/` và `no_helmet/`

### Bước 2: Preprocessing dữ liệu

```bash
cd helmet_detection_project
python data_preprocessing/preprocess_data.py
```

**Lưu ý:** Nếu file này không tồn tại, preprocessing sẽ tự động chạy trong bước training.

### Bước 3: Training model

**Training nhanh (để test):**
```bash
cd helmet_detection_project
python training/train_model.py --mode quick --epochs 20
```

**Training đầy đủ (Khuyến nghị):**
```bash
cd helmet_detection_project
python training/train_model.py --mode full --epochs 50 --batch_size 32
```

**Training với transfer learning (ResNet50 - Accuracy cao):**
```bash
cd helmet_detection_project
python training/train_model.py --mode transfer --epochs 30
```

**Training với MobileNetV2 (Nhẹ, phù hợp mobile):**
```bash
cd helmet_detection_project
python training/train_model.py --mode mobile --epochs 40
```

### Bước 4: Kiểm tra kết quả training

Model sẽ được lưu tại:
- `helmet_detection_project/models/helmet_detection_model.h5`
- `helmet_detection_project/models/best_helmet_model.h5`

Kết quả training:
- `helmet_detection_project/models/training_history.png`
- `helmet_detection_project/models/confusion_matrix.png`

---

## 🖥️ Hướng dẫn sử dụng ứng dụng

### Giao diện chính

Ứng dụng có giao diện đồ họa với các nút:

1. **Bắt đầu Camera** / **Nhận diện từ Camera**
   - Bắt đầu nhận diện từ webcam
   - Hiển thị kết quả real-time
   - Tự động chụp và lưu ảnh khi phát hiện

2. **Nhận diện từ Ảnh**
   - Chọn file ảnh từ máy tính
   - Xử lý và hiển thị kết quả
   - Lưu ảnh kết quả vào `results/`

3. **Nhận diện từ Video**
   - Chọn file video
   - Xử lý từng frame
   - Lưu video kết quả vào `results/`

4. **Dừng Camera**
   - Dừng chế độ camera

### Thống kê mũ bảo hiểm (Chỉ có trong `head_detection_app_with_helmet.py`)

- **Tổng phát hiện**: Số lượng đầu người đã phát hiện
- **Có mũ bảo hiểm**: Số lượng có mũ (màu xanh)
- **Không có mũ**: Số lượng không có mũ (màu đỏ)

### Thư mục lưu trữ

- **Ảnh/video kết quả**: `results/`
- **Ảnh đầu người đã chụp**: `captured_heads/session_YYYYMMDD_HHMMSS/`

---

## ⚠️ Xử lý lỗi thường gặp

### 1. Lỗi "No module named 'tensorflow'"

**Giải pháp:**
```bash
pip install tensorflow==2.13.0
```

### 2. Lỗi "No module named 'ultralytics'"

**Giải pháp:**
```bash
pip install -r requirements.txt
```

### 3. Lỗi "Không thể mở camera"

**Nguyên nhân:**
- Camera đang được sử dụng bởi ứng dụng khác
- Camera không được kết nối
- Index camera không đúng

**Giải pháp:**
- Đóng các ứng dụng khác đang dùng camera (Skype, Zoom, Teams, etc.)
- Kiểm tra camera đã được kết nối
- Thử đổi camera index trong code (0 → 1 hoặc 1 → 0)

### 4. Lỗi "Model nhận diện mũ bảo hiểm chưa tồn tại"

**Giải pháp:**
- Training model trước (xem phần Training bên trên)
- Hoặc sử dụng `python apps/copy_helmet_model.py` để copy model nếu có

### 5. Ứng dụng chạy chậm

**Giải pháp:**
- Sử dụng GPU NVIDIA với CUDA:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- Giảm độ phân giải camera trong code
- Sử dụng model nhỏ hơn (yolov8n.pt thay vì yolov8l.pt)

### 6. Lỗi "Permission denied" khi lưu file

**Giải pháp:**
- Kiểm tra quyền ghi vào thư mục
- Chạy terminal/command prompt với quyền Administrator (Windows)
- Chạy với sudo (Linux/Mac)

### 7. Import error khi chạy từ apps folder

**Giải pháp:**
- Đảm bảo đang chạy từ thư mục root của dự án:
  ```bash
  cd C:\Python\DuAnHocMay\NHANDIENMATNGUOI_YOLO
  python apps/head_detection_app_with_helmet.py
  ```
- Không chạy từ trong folder `apps/`:
  ```bash
  # ❌ SAI
  cd apps
  python head_detection_app_with_helmet.py
  
  # ✅ ĐÚNG
  python apps/head_detection_app_with_helmet.py
  ```

---

## 📝 Checklist chạy dự án

### Lần đầu chạy:

- [ ] Đã cài đặt Python 3.8+
- [ ] Đã tạo và kích hoạt virtual environment
- [ ] Đã cài đặt `requirements.txt`
- [ ] Đã cài đặt TensorFlow
- [ ] Đã test model YOLO (`python apps/demo_test.py`)
- [ ] Đã có model helmet (nếu muốn dùng helmet detection)
- [ ] Đã kiểm tra camera hoạt động

### Chạy ứng dụng:

- [ ] Đang ở thư mục root của dự án
- [ ] Virtual environment đã được kích hoạt
- [ ] Đã chọn ứng dụng phù hợp (có/không có helmet detection)
- [ ] Đã kiểm tra thư mục `results/` và `captured_heads/` có quyền ghi

---

## 🔧 Tùy chỉnh nâng cao

### Thay đổi model YOLO

Mở file `apps/head_detection_app.py` (hoặc `apps/head_detection_app_with_helmet.py`), tìm dòng:
```python
self.model = YOLO('yolov8n.pt')
```

Thay đổi thành:
- `yolov8n.pt` - Nano (nhanh nhất, ít chính xác nhất)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large (chậm nhất, chính xác nhất)
- `yolov8x.pt` - XLarge

### Thay đổi độ tin cậy (Confidence)

Tìm dòng:
```python
results = self.model(frame, classes=[0], conf=0.5)
```

Thay đổi `0.5` thành giá trị khác (0.0 - 1.0):
- `0.3` - Phát hiện nhiều hơn (có thể có false positives)
- `0.5` - Mặc định
- `0.7` - Phát hiện chắc chắn hơn (có thể miss một số)

---

## 💡 Mẹo sử dụng

1. **Lần đầu chạy**: Nên chạy `python apps/demo_test.py` trước để đảm bảo YOLO hoạt động

2. **Training model**: Nếu dataset nhỏ (< 100 ảnh mỗi class), dùng `--mode quick` với ít epochs

3. **Lưu kết quả**: Tất cả kết quả sẽ tự động lưu vào `results/` - không cần lo lắng về việc mất dữ liệu

4. **Camera**: Nếu camera không hoạt động, thử đổi camera index hoặc kiểm tra quyền truy cập

5. **Performance**: Nếu máy yếu, dùng `yolov8n.pt` và giảm độ phân giải camera

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra lại các bước cài đặt
2. Xem phần "Xử lý lỗi thường gặp"
3. Kiểm tra README.md để biết thêm chi tiết
4. Xem logs trong ứng dụng để biết lỗi cụ thể

---

**Chúc bạn sử dụng dự án thành công! 🎉**

