"""
Module xử lý nhận diện khuôn mặt và đầu người
Tách logic detection từ main app để dễ bảo trì và mở rộng
"""

import cv2
import numpy as np
from ultralytics import YOLO


class FaceHeadDetector:
    """Class chuyên xử lý nhận diện khuôn mặt và đầu người"""
    
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self._load_models()
    
    def _load_models(self):
        """Load model YOLO và face cascade"""
        try:
            # Load YOLO model
            self.model = YOLO('yolov8n.pt')
            
            # Load Haar Cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            print("✅ Đã tải thành công các model nhận diện!")
            
        except Exception as e:
            print(f"❌ Lỗi khi tải model: {str(e)}")
            raise e
    
    def detect_faces_and_heads(self, frame):
        """
        Kết hợp phát hiện khuôn mặt và đầu người với cải tiến
        
        Args:
            frame: Frame ảnh đầu vào
            
        Returns:
            tuple: (annotated_frame, detection_count, detected_regions)
        """
        if self.model is None or self.face_cascade is None:
            return frame, 0, []
        
        detected_regions = []
        
        # 1. Phát hiện khuôn mặt bằng Haar Cascade
        faces = self._detect_faces_haar(frame)
        
        # Vẽ bounding box cho khuôn mặt hợp lệ
        for (x, y, w, h) in faces:
            # Mở rộng để bao gồm cả đầu
            margin_x = int(w * 0.3)
            margin_y = int(h * 0.4)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Face+Head', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            detected_regions.append((x1, y1, x2-x1, y2-y1))
        
        # 2. Phát hiện đầu người bằng YOLO (nếu không có khuôn mặt hợp lệ)
        if len(faces) == 0:
            heads = self._detect_heads_yolo(frame)
            
            for (x1, y1, w, h) in heads:
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Head', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                detected_regions.append((x1, y1, w, h))
        
        return frame, len(detected_regions), detected_regions
    
    def _detect_faces_haar(self, frame):
        """Phát hiện khuôn mặt bằng Haar Cascade với tham số tối ưu"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Cải tiến: Thêm histogram equalization để tăng độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=12,
            minSize=(80, 80),
            maxSize=(250, 250),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Lọc kết quả để giảm false positive
        valid_faces = []
        for (x, y, w, h) in faces:
            # Kiểm tra tỷ lệ khung hình hợp lý
            aspect_ratio = w / h
            if 0.7 <= aspect_ratio <= 1.4:
                # Kiểm tra vùng trung tâm có đủ tương phản không
                face_roi = gray[y:y+h, x:x+w]
                if face_roi.size > 0:
                    mean_intensity = np.mean(face_roi)
                    std_intensity = np.std(face_roi)
                    # Khuôn mặt thường có độ tương phản cao
                    if std_intensity > 20 and 50 < mean_intensity < 200:
                        valid_faces.append((x, y, w, h))
        
        return valid_faces
    
    def _detect_heads_yolo(self, frame):
        """Phát hiện đầu người bằng YOLO"""
        results = self.model(frame, classes=[0], conf=0.5)
        boxes = results[0].boxes
        
        detected_heads = []
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # Kiểm tra kích thước hợp lý
                height = y2 - y1
                width = x2 - x1
                
                # Chỉ chấp nhận bounding box có kích thước hợp lý
                if height > 100 and width > 50:
                    head_height = height * 0.5
                    
                    head_x1 = max(0, x1 + width * 0.05)
                    head_x2 = min(frame.shape[1], x2 - width * 0.05)
                    head_y1 = max(0, y1 + height * 0.02)
                    head_y2 = min(frame.shape[0], y1 + head_height)
                    
                    detected_heads.append((int(head_x1), int(head_y1), 
                                          int(head_x2 - head_x1), int(head_y2 - head_y1)))
        
        return detected_heads
    
    def is_model_loaded(self):
        """Kiểm tra xem model đã được load chưa"""
        return self.model is not None and self.face_cascade is not None
