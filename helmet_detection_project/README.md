# Hướng dẫn sử dụng hệ thống nhận diện mũ bảo hiểm

## 📋 Tổng quan

Hệ thống nhận diện mũ bảo hiểm sử dụng Deep Learning (CNN) để phân loại người có đội mũ bảo hiểm hay không. Hệ thống bao gồm:

1. **Thu thập dữ liệu** - Chụp và phân loại ảnh thủ công
2. **Preprocessing** - Xử lý và chuẩn bị dữ liệu
3. **Training** - Train model CNN
4. **Inference** - Sử dụng model để dự đoán real-time

## 🚀 Cài đặt

### Bước 1: Cài đặt dependencies
```bash
cd helmet_detection_project
pip install -r requirements.txt
```

### Bước 2: Kiểm tra cài đặt
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

## 📊 Quy trình sử dụng

### **Bước 1: Thu thập dữ liệu**

#### Chạy ứng dụng thu thập dữ liệu:
```bash
python data_collection/data_collection_app.py
```

#### Hướng dẫn sử dụng:
1. **Bắt đầu Camera** - Mở camera để chụp ảnh
2. **Chụp ảnh có mũ bảo hiểm** - Nhấn nút "Chụp - Có mũ bảo hiểm" khi người đội mũ
3. **Chụp ảnh không có mũ** - Nhấn nút "Chụp - Không có mũ" khi người không đội mũ
4. **Thu thập đủ dữ liệu** - Khuyến nghị ít nhất 200-500 ảnh mỗi loại

#### Cấu trúc dữ liệu:
```
data_collection/
├── with_helmet/          # Ảnh có mũ bảo hiểm
│   ├── helmet_20250126_143052_123.jpg
│   └── ...
└── no_helmet/            # Ảnh không có mũ
    ├── no_helmet_20250126_143052_456.jpg
    └── ...
```

### **Bước 2: Preprocessing dữ liệu**

#### Chạy script preprocessing:
```bash
python data_preprocessing/preprocess_data.py
```

#### Chức năng:
- Load và resize ảnh về kích thước chuẩn (224x224)
- Normalize pixel values về [0, 1]
- Chia dữ liệu thành train/validation/test (60%/20%/20%)
- Data augmentation (tùy chọn)
- Lưu dữ liệu đã xử lý

#### Kết quả:
```
data_preprocessing/
├── X_train.npy          # Training images
├── X_val.npy           # Validation images
├── X_test.npy          # Test images
├── y_train.npy         # Training labels
├── y_val.npy           # Validation labels
├── y_test.npy          # Test labels
├── metadata.pkl        # Metadata
└── data_distribution.png  # Biểu đồ phân bố
```

### **Bước 3: Training model**

#### Chạy training:
```bash
# Training nhanh (test)
python training/train_model.py --mode quick --epochs 20

# Training đầy đủ
python training/train_model.py --mode full --epochs 50

# Training với transfer learning
python training/train_model.py --mode transfer --epochs 30

# Training model mobile
python training/train_model.py --mode mobile --epochs 40
```

#### Các mode training:

| Mode | Mô tả | Ưu điểm | Nhược điểm |
|------|-------|----------|------------|
| **quick** | Custom CNN, ít epochs | Nhanh, test nhanh | Accuracy thấp |
| **full** | Custom CNN, đầy đủ | Cân bằng tốt | Thời gian trung bình |
| **transfer** | ResNet50 pre-trained | Accuracy cao | Chậm, cần GPU |
| **mobile** | MobileNetV2 | Nhẹ, phù hợp mobile | Accuracy trung bình |

#### Kết quả training:
```
models/
├── helmet_detection_model.h5    # Model đã train
├── best_helmet_model.h5         # Model tốt nhất
├── training_results.pkl         # Kết quả training
├── training_history.png         # Đồ thị training
└── confusion_matrix.png         # Confusion matrix
```

### **Bước 4: Inference (Sử dụng model)**

#### Chạy ứng dụng inference:
```bash
python inference/inference_app.py
```

#### Chức năng:
- **Camera real-time** - Nhận diện mũ bảo hiểm từ camera
- **Test từ ảnh** - Upload ảnh để test
- **Thống kê** - Đếm số lần phát hiện có/không có mũ
- **Confidence score** - Độ tin cậy của dự đoán

#### Giao diện:
- **Camera Feed** - Hiển thị video với bounding box
- **Thống kê** - Số lượng phát hiện theo loại
- **Kết quả hiện tại** - Phân loại tổng thể
- **Log** - Lịch sử các phát hiện

## 🎯 Tips để có kết quả tốt

### **Thu thập dữ liệu:**
- ✅ Chụp nhiều góc độ khác nhau
- ✅ Đa dạng ánh sáng (sáng, tối, trong nhà, ngoài trời)
- ✅ Đa dạng loại mũ bảo hiểm
- ✅ Cân bằng số lượng ảnh có/không có mũ
- ✅ Chất lượng ảnh tốt, không bị mờ

### **Training:**
- ✅ Sử dụng GPU nếu có thể
- ✅ Monitor training loss và accuracy
- ✅ Sử dụng early stopping để tránh overfitting
- ✅ Thử các model khác nhau (custom, ResNet, MobileNet)
- ✅ Data augmentation để tăng dữ liệu

### **Inference:**
- ✅ Đảm bảo ánh sáng đủ
- ✅ Camera chất lượng tốt
- ✅ Người trong khung hình rõ ràng
- ✅ Kiểm tra confidence score

## 🔧 Troubleshooting

### **Lỗi thường gặp:**

#### 1. "Model chưa được load"
```bash
# Kiểm tra file model có tồn tại không
ls models/helmet_detection_model.h5

# Nếu không có, chạy training trước
python training/train_model.py --mode quick
```

#### 2. "Không thể mở camera"
```bash
# Kiểm tra camera có hoạt động không
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"

# Thử camera khác
# Trong code, thay cv2.VideoCapture(0) thành cv2.VideoCapture(1)
```

#### 3. "Không tìm thấy dữ liệu"
```bash
# Kiểm tra thư mục dữ liệu
ls data_collection/with_helmet/
ls data_collection/no_helmet/

# Nếu trống, chạy data collection trước
python data_collection/data_collection_app.py
```

#### 4. "Out of memory" khi training
```bash
# Giảm batch size
python training/train_model.py --mode full --batch_size 16

# Hoặc sử dụng model nhẹ hơn
python training/train_model.py --mode mobile
```

## 📈 Đánh giá hiệu suất

### **Metrics quan trọng:**
- **Accuracy** - Độ chính xác tổng thể
- **Precision** - Độ chính xác khi dự đoán "có mũ"
- **Recall** - Tỷ lệ phát hiện đúng "có mũ"
- **F1-Score** - Cân bằng precision và recall

### **Confusion Matrix:**
```
                Predicted
Actual    No Helmet  With Helmet
No Helmet     TN         FP
With Helmet   FN         TP
```

### **Cải thiện hiệu suất:**
1. **Tăng dữ liệu** - Thu thập thêm ảnh
2. **Data augmentation** - Tăng cường dữ liệu
3. **Transfer learning** - Sử dụng model pre-trained
4. **Hyperparameter tuning** - Điều chỉnh learning rate, batch size
5. **Ensemble methods** - Kết hợp nhiều model

## 🚀 Mở rộng

### **Tính năng có thể thêm:**
- [ ] Nhận diện nhiều người cùng lúc
- [ ] Phân loại loại mũ bảo hiểm
- [ ] Phát hiện mũ không đúng cách
- [ ] Tích hợp với hệ thống giám sát
- [ ] Mobile app
- [ ] Web interface
- [ ] API service

### **Cải tiến model:**
- [ ] YOLO object detection
- [ ] Transformer architecture
- [ ] Multi-task learning
- [ ] Real-time optimization
- [ ] Edge deployment

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. **Log files** - Xem thông báo lỗi chi tiết
2. **Dependencies** - Đảm bảo đã cài đặt đầy đủ
3. **Data quality** - Kiểm tra chất lượng dữ liệu
4. **Hardware** - Đảm bảo camera và GPU hoạt động

---

**Chúc bạn thành công với dự án nhận diện mũ bảo hiểm! 🎉**


