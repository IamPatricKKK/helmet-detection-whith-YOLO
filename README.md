# Ứng dụng Nhận diện Khuôn mặt + Mũ Bảo Hiểm

## Mô tả dự án

Ứng dụng nhận diện khuôn mặt và mũ bảo hiểm sử dụng:
- **Haar Cascade** để nhận diện khuôn mặt
- **CNN Model** để phân loại có/không có mũ bảo hiểm
- Giao diện **Tkinter** đơn giản, dễ sử dụng

## Tính năng chính

- ✅ **Nhận diện từ Ảnh**: Upload ảnh và nhận diện khuôn mặt, tự động phân loại có mũ/không có mũ
- ✅ **Nhận diện từ Camera**: Nhận diện real-time và tự động chụp ảnh mỗi 2 giây khi phát hiện khuôn mặt
- ✅ **Tự động phân loại**: Ảnh được lưu vào folder `with_helmet/` hoặc `without_helmet/` tương ứng
- ✅ **Hiển thị kết quả**: Viền xanh (YES) = Có mũ bảo hiểm, Viền đỏ (NO) = Không có mũ

## Yêu cầu hệ thống

### Phần mềm cần thiết
- **Python 3.8+**: [Tải Python](https://www.python.org/downloads/)
- **Git** (tùy chọn): [Tải Git](https://git-scm.com/downloads)

### Phần cứng khuyến nghị
- **RAM**: Tối thiểu 4GB, khuyến nghị 8GB+
- **GPU**: NVIDIA GPU với CUDA (tùy chọn, để tăng tốc độ)
- **Camera**: Webcam hoặc camera USB (cho chế độ camera)

## Hướng dẫn cài đặt

### Bước 1: Clone hoặc tải dự án
```bash
# Nếu sử dụng Git
git clone <repository-url>
cd NHANDIENMATNGUOI_YOLO

# Hoặc tải file ZIP và giải nén
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)
```bash
# Tạo môi trường ảo trong folder apps
cd apps
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
# Cài đặt từ requirements.txt
pip install -r requirements.txt

# Hoặc cài đặt thủ công
pip install opencv-python ultralytics pillow numpy tensorflow
```

**Lưu ý**: 
- Lần đầu chạy, YOLO sẽ tự động tải model `yolov8n.pt` (khoảng 6MB)
- Model mũ bảo hiểm đã được chuẩn bị sẵn trong `apps/model/best_model.h5`

### Bước 4: Chạy ứng dụng

**Cách 1: Sử dụng batch file (Windows)**
```bash
run_helmet_detection.bat
```

**Cách 2: Chạy trực tiếp từ Python**
```bash
cd apps
python head_detection_app_with_helmet.py
```

**Cách 3: Chạy với venv Python**
```bash
cd apps
.\venv\Scripts\python.exe head_detection_app_with_helmet.py
```

## Cấu trúc dự án

```
NHANDIENMATNGUOI_YOLO/
├── apps/                              # Folder chứa ứng dụng chính
│   ├── head_detection_app_with_helmet.py  # ⭐ Ứng dụng chính
│   ├── model/                         # Model mũ bảo hiểm
│   │   ├── best_model.h5             # Model chính
│   │   └── final_model.h5             # Model backup
│   ├── yolov8n.pt                     # Model YOLO v8
│   ├── captured_heads/                # Ảnh đã chụp (tự động tạo)
│   │   └── session_XXX/
│   │       ├── with_helmet/          # Ảnh có mũ
│   │       └── without_helmet/       # Ảnh không có mũ
│   ├── results/                       # Kết quả nhận diện từ ảnh
│   └── venv/                          # Môi trường ảo (nếu tạo)
├── helmet_detection_project/          # Model source (nếu cần train lại)
│   └── models/
│       ├── best_model.h5
│       └── final_model.h5
├── requirements.txt                   # Danh sách thư viện
├── README.md                          # Tài liệu này
├── run_helmet_detection.bat          # Batch file chạy ứng dụng (Windows)
└── yolov8n.pt                        # Model YOLO (backup)
```

## Hướng dẫn sử dụng

### Giao diện chính
Ứng dụng có giao diện đồ họa với các nút:
- **Bắt đầu Camera**: Bắt đầu nhận diện từ webcam
- **Nhận diện từ Ảnh**: Chọn và xử lý file ảnh
- **Dừng Camera**: Dừng chế độ camera

### Chế độ Camera
1. Nhấn "Bắt đầu Camera"
2. Camera sẽ bắt đầu và hiển thị kết quả nhận diện
3. Ảnh sẽ được tự động chụp mỗi 2 giây khi phát hiện khuôn mặt
4. Ảnh được tự động phân loại và lưu vào folder tương ứng:
   - `captured_heads/session_XXX/with_helmet/` - Có mũ bảo hiểm
   - `captured_heads/session_XXX/without_helmet/` - Không có mũ
5. Nhấn "Dừng Camera" để dừng

### Chế độ Ảnh
1. Nhấn "Nhận diện từ Ảnh"
2. Chọn file ảnh (hỗ trợ: jpg, jpeg, png, bmp, tiff)
3. Kết quả sẽ hiển thị với bounding boxes:
   - **Viền XANH + "YES"** = Có mũ bảo hiểm
   - **Viền ĐỎ + "NO"** = Không có mũ bảo hiểm
4. Tất cả khuôn mặt phát hiện được sẽ tự động được lưu và phân loại

## Luồng hoạt động

```
Camera Frame / Image
  ↓
Nhận diện khuôn mặt (Haar Cascade)
  ↓
Với mỗi khuôn mặt:
  ├─ Predict mũ bảo hiểm (CNN Model từ apps/model/)
  ├─ Vẽ bounding box + text (Xanh/Đỏ)
  └─ Chụp ảnh mỗi 2 giây → Lưu vào folder tương ứng
```

## Các thư viện được sử dụng

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| ultralytics | 8.0.196 | Mô hình YOLO v8 |
| opencv-python | 4.8.1.78 | Xử lý hình ảnh và video |
| Pillow | 10.0.1 | Xử lý ảnh |
| numpy | 1.24.3 | Tính toán số học |
| tensorflow | 2.13.0 | Framework cho CNN model |
| torch | 2.0.1 | Deep learning framework |
| torchvision | 0.15.2 | Computer vision utilities |
| matplotlib | 3.7.2 | Vẽ biểu đồ |

## Xử lý sự cố

### Lỗi thường gặp

#### 1. "ModuleNotFoundError: No module named 'cv2'"
```bash
# Cài đặt OpenCV
pip install opencv-python
```

#### 2. "ModuleNotFoundError: No module named 'tensorflow'"
```bash
# Cài đặt TensorFlow
pip install tensorflow
```

#### 3. "Không thể mở camera"
- Kiểm tra camera có được kết nối không
- Đảm bảo không có ứng dụng khác đang sử dụng camera
- Thử thay đổi camera index trong code (0 → 1)

#### 4. "Model nhận diện mũ bảo hiểm chưa tồn tại"
- Kiểm tra file `apps/model/best_model.h5` có tồn tại không
- Đảm bảo model được đặt đúng vị trí

#### 5. Hiệu suất chậm
- Sử dụng GPU NVIDIA với CUDA
- Giảm độ phân giải camera
- Sử dụng model nhỏ hơn (yolov8n thay vì yolov8l)

### Tối ưu hiệu suất

#### Sử dụng GPU
```bash
# Cài đặt PyTorch với CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Cấu hình

### Thay đổi thư mục lưu ảnh
Trong giao diện, nhấn "Đổi thư mục" để chọn thư mục lưu ảnh khác.

### Thay đổi thời gian chụp ảnh
Trong file `head_detection_app_with_helmet.py`, dòng 48:
```python
self.capture_interval = 2.0  # Chụp mỗi 2 giây (có thể thay đổi)
```

### Thay đổi độ tin cậy detection
Trong file `head_detection_app_with_helmet.py`, trong hàm `detectMultiScale`:
```python
minNeighbors=12,  # Tăng giá trị = ít false positive hơn
minSize=(80, 80),  # Kích thước khuôn mặt tối thiểu
```

## Liên hệ và hỗ trợ

Nếu gặp vấn đề hoặc có câu hỏi, vui lòng:
1. Kiểm tra phần "Xử lý sự cố" trong tài liệu này
2. Tạo issue trên repository
3. Liên hệ qua email

## Giấy phép

Dự án này sử dụng các thư viện mã nguồn mở. Vui lòng tuân thủ các giấy phép của từng thư viện.

---

**Lưu ý**: 
- Dự án này nhận diện khuôn mặt nói chung, không phải nhận diện khuôn mặt cụ thể
- Model mũ bảo hiểm đã được train sẵn và đặt trong `apps/model/best_model.h5`
- Nếu muốn train lại model, có thể tham khảo code trong `helmet_detection_project/`
