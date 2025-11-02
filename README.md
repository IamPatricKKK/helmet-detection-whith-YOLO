# Ứng dụng Nhận diện Khuôn mặt sử dụng YOLO v8 và MTCNN

## Mô tả dự án

Dự án này cung cấp nhiều phương pháp để nhận diện khuôn mặt với các tính năng:
- **Nhận diện từ Camera trực tiếp**: Sử dụng webcam để nhận diện khuôn mặt trong thời gian thực
- **Nhận diện từ Ảnh**: Upload và xử lý ảnh để tìm khuôn mặt
- **Nhận diện từ Video**: Xử lý file video để nhận diện khuôn mặt trong từng frame

## Các phiên bản ứng dụng

### 1. `face_detection_app.py` - Phiên bản cơ bản
- Sử dụng YOLO v8 để nhận diện toàn bộ người
- Phù hợp cho nhận diện người nói chung

### 2. `face_detection_app_v2.py` - Phiên bản nhận diện khuôn mặt với tự động chụp ảnh ⭐
- Sử dụng Haar Cascade để nhận diện khuôn mặt
- **Tự động chụp và lưu khuôn mặt** khi phát hiện khuôn mặt mới
- Có thể chọn thư mục lưu ảnh
- Tránh chụp trùng lặp với thuật toán thông minh
- Hiển thị thông tin chi tiết về việc chụp ảnh

### 3. `advanced_face_detection_app.py` - Phiên bản nâng cao
- Hỗ trợ cả Haar Cascade và MTCNN
- MTCNN cho độ chính xác cao hơn với landmarks
- Có thể chọn phương pháp nhận diện

### 4. `head_detection_app.py` - Phiên bản nhận diện đầu người ⭐
- Sử dụng YOLO v8 để nhận diện đầu người
- Thuật toán thông minh để tính toán vùng đầu chính xác
- Tự động chụp và lưu đầu người khi phát hiện
- Bao gồm cả đầu và cổ trong bounding box
- Hiển thị điểm trung tâm và độ tin cậy

### 5. `advanced_head_detection_app.py` - Phiên bản nhận diện đầu người nâng cao
- Thuật toán nâng cao để nhận diện đầu người chính xác hơn
- Mở rộng bounding box để bao gồm cả tóc và cổ
- Hiển thị điểm trung tâm của đầu


## Yêu cầu hệ thống

### Phần mềm cần thiết
- **Python 3.8+**: [Tải Python](https://www.python.org/downloads/)
- **Git** (tùy chọn): [Tải Git](https://git-scm.com/downloads)

### Phần cứng khuyến nghị
- **RAM**: Tối thiểu 4GB, khuyến nghị 8GB+
- **GPU**: NVIDIA GPU với CUDA (tùy chọn, để tăng tốc độ)
- **Camera**: Webcam hoặc camera USB (cho chế độ camera trực tiếp)

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
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết

**Cho phiên bản cơ bản:**
```bash
pip install -r requirements.txt
```

**Cho phiên bản nâng cao (MTCNN):**
```bash
pip install -r requirements_v2.txt
```

**Lưu ý**: 
- Lần đầu chạy, YOLO sẽ tự động tải model `yolov8n.pt` (khoảng 6MB)
- MTCNN sẽ tải các model weights khi chạy lần đầu

### Bước 4: Chạy ứng dụng

**Phiên bản cơ bản (nhận diện người):**
```bash
python face_detection_app.py
```

**Phiên bản nhận diện khuôn mặt với tự động chụp ảnh (Khuyến nghị):**
```bash
python face_detection_app_v2.py
```

**Phiên bản nâng cao (Haar + MTCNN):**
```bash
python advanced_face_detection_app.py
```

**Phiên bản nhận diện đầu người (Khuyến nghị):**
```bash
python head_detection_app.py
```

**Phiên bản nhận diện đầu người nâng cao:**
```bash
python advanced_head_detection_app.py
```


## Các thư viện được sử dụng

### Phiên bản cơ bản (requirements.txt)
| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| ultralytics | 8.0.196 | Mô hình YOLO v8 |
| opencv-python | 4.8.1.78 | Xử lý hình ảnh và video |
| Pillow | 10.0.1 | Xử lý ảnh |
| numpy | 1.24.3 | Tính toán số học |
| torch | 2.0.1 | Deep learning framework |
| torchvision | 0.15.2 | Computer vision utilities |
| matplotlib | 3.7.2 | Vẽ biểu đồ |
| tkinter-tooltip | 2.0.0 | Tooltip cho giao diện |

### Phiên bản nâng cao (requirements_v2.txt)
| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| mtcnn | 0.1.1 | Nhận diện khuôn mặt với landmarks |
| tensorflow | 2.13.0 | Framework cho MTCNN |
| *(các thư viện khác giống phiên bản cơ bản)* | | |

## Hướng dẫn sử dụng

### Giao diện chính
Ứng dụng có giao diện đồ họa với các nút:
- **Nhận diện từ Camera**: Bắt đầu nhận diện từ webcam
- **Nhận diện từ Ảnh**: Chọn và xử lý file ảnh
- **Nhận diện từ Video**: Chọn và xử lý file video
- **Dừng Camera**: Dừng chế độ camera

### Chế độ Camera
1. Nhấn "Nhận diện từ Camera"
2. Camera sẽ bắt đầu và hiển thị kết quả nhận diện
3. Số lượng người được phát hiện sẽ hiển thị trong phần thông tin
4. Nhấn "Dừng Camera" để dừng

### Chế độ Ảnh
1. Nhấn "Nhận diện từ Ảnh"
2. Chọn file ảnh (hỗ trợ: jpg, jpeg, png, bmp, tiff)
3. Kết quả sẽ hiển thị với bounding boxes
4. Ảnh kết quả sẽ được lưu với prefix "result_"

### Chế độ Video
1. Nhấn "Nhận diện từ Video"
2. Chọn file video (hỗ trợ: mp4, avi, mov, mkv, wmv)
3. Video sẽ được xử lý frame by frame
4. Video kết quả sẽ được lưu với prefix "result_"

## Xử lý sự cố

### Lỗi thường gặp

#### 1. "Không thể mở camera"
- Kiểm tra camera có được kết nối không
- Đảm bảo không có ứng dụng khác đang sử dụng camera
- Thử thay đổi camera index trong code (0 → 1)

#### 2. "Model chưa được tải"
- Kiểm tra kết nối internet để tải model
- Đảm bảo đã cài đặt đầy đủ các thư viện

#### 3. "Không thể đọc file ảnh/video"
- Kiểm tra định dạng file có được hỗ trợ không
- Đảm bảo file không bị hỏng

#### 4. Hiệu suất chậm
- Sử dụng GPU NVIDIA với CUDA
- Giảm độ phân giải video/ảnh
- Sử dụng model nhỏ hơn (yolov8n thay vì yolov8l)

### Tối ưu hiệu suất

#### Sử dụng GPU
```bash
# Cài đặt PyTorch với CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Thay đổi model
Trong file `face_detection_app.py`, dòng 45:
```python
# Model nhỏ (nhanh, ít chính xác)
self.model = YOLO('yolov8n.pt')

# Model lớn (chậm, chính xác hơn)
self.model = YOLO('yolov8l.pt')
```

## Cấu trúc dự án

```
NHANDIENMATNGUOI_YOLO/
├── face_detection_app.py    # Ứng dụng chính
├── requirements.txt         # Danh sách thư viện
├── README.md               # Tài liệu này
├── result_*.jpg           # Ảnh kết quả (tự tạo)
├── result_*.mp4           # Video kết quả (tự tạo)
└── venv/                  # Môi trường ảo (nếu tạo)
```

## Tính năng nâng cao

### Tùy chỉnh độ tin cậy
Trong code, bạn có thể thay đổi threshold:
```python
results = self.model(image, classes=[0], conf=0.5)  # conf=0.5 là 50%
```

### Lưu kết quả chi tiết
```python
# Lưu kết quả với thông tin chi tiết
results[0].save('detailed_result.jpg')
```

### Nhận diện nhiều class
```python
# Nhận diện người và xe
results = self.model(image, classes=[0, 2])  # 0=person, 2=car
```

## Liên hệ và hỗ trợ

Nếu gặp vấn đề hoặc có câu hỏi, vui lòng:
1. Kiểm tra phần "Xử lý sự cố" trong tài liệu này
2. Tạo issue trên repository
3. Liên hệ qua email

## Giấy phép

Dự án này sử dụng các thư viện mã nguồn mở. Vui lòng tuân thủ các giấy phép của từng thư viện.

---

**Lưu ý**: Dự án này chỉ nhận diện người nói chung, không phải nhận diện khuôn mặt cụ thể. Để nhận diện khuôn mặt cụ thể, bạn cần sử dụng các mô hình chuyên biệt như FaceNet hoặc ArcFace.
