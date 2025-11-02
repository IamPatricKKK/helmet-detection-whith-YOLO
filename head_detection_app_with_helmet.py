"""
Ứng dụng Nhận diện Khuôn mặt & Đầu người với tính năng nhận diện mũ bảo hiểm
Tích hợp model CNN để phân loại có/không có mũ bảo hiểm
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
import time
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
# import tensorflow as tf


class HeadDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Nhận diện Khuôn mặt & Đầu người + Mũ bảo hiểm")
        self.root.geometry("1000x700")
        
        # Khởi tạo model YOLO và face cascade
        self.model = None
        self.face_cascade = None
        self.helmet_model = None  # Model nhận diện mũ bảo hiểm
        self.camera_running = False
        self.cap = None
        
        # Tạo thư mục lưu ảnh với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_folder = "captured_heads"
        self.save_folder = os.path.join(self.main_folder, f"session_{timestamp}")
        
        # Tạo thư mục chính nếu chưa có
        if not os.path.exists(self.main_folder):
            os.makedirs(self.main_folder)
        
        # Tạo thư mục session
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        
        # Biến để theo dõi trạng thái phát hiện và chụp ảnh
        self.last_detection_count = 0
        self.last_capture_time = 0
        self.stability_wait_time = 2.0
        self.detection_start_time = 0
        self.captured_faces_history = []
        self.last_no_detection_log = 0
        
        # Biến thống kê mũ bảo hiểm
        self.helmet_stats = {
            'total_detections': 0,
            'with_helmet': 0,
            'without_helmet': 0
        }
        
        # Tạo giao diện
        self.create_widgets()
        
        # Load models
        self.load_models()
        
        # Thông báo thư mục mới
        self.log_info(f"📁 Thư mục lưu ảnh: {self.save_folder}")
    
    def create_widgets(self):
        """Tạo giao diện"""
        # Frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tiêu đề
        title_label = ttk.Label(main_frame, text="Ứng dụng Nhận diện Khuôn mặt & Đầu người + Mũ bảo hiểm", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Frame chọn chế độ
        mode_frame = ttk.LabelFrame(main_frame, text="Chọn chế độ nhận diện", padding="10")
        mode_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Buttons
        self.camera_btn = ttk.Button(mode_frame, text="Bắt đầu Camera", 
                                   command=self.start_camera_detection)
        self.camera_btn.grid(row=0, column=0, padx=5)
        
        self.image_btn = ttk.Button(mode_frame, text="Nhận diện từ Ảnh", 
                                   command=self.detect_from_image)
        self.image_btn.grid(row=0, column=1, padx=5)
        
        self.video_btn = ttk.Button(mode_frame, text="Nhận diện từ Video", 
                                   command=self.detect_from_video)
        self.video_btn.grid(row=0, column=2, padx=5)
        
        self.stop_btn = ttk.Button(mode_frame, text="Dừng Camera", 
                                  command=self.stop_camera, state="disabled")
        self.stop_btn.grid(row=0, column=3, padx=5)
        
        # Frame cài đặt
        settings_frame = ttk.LabelFrame(main_frame, text="Cài đặt", padding="10")
        settings_frame.grid(row=1, column=2, sticky=(tk.W, tk.E), padx=(10, 0))
        
        ttk.Label(settings_frame, text="Thư mục lưu:").grid(row=0, column=0, padx=5)
        self.folder_label = ttk.Label(settings_frame, text=self.save_folder, foreground="blue")
        self.folder_label.grid(row=0, column=1, padx=5)
        
        ttk.Button(settings_frame, text="Đổi thư mục", 
                  command=self.change_save_folder).grid(row=0, column=2, padx=5)
        
        ttk.Button(settings_frame, text="Reset lịch sử", 
                  command=self.reset_captured_history).grid(row=1, column=0, columnspan=3, pady=5)
        
        # Frame hiển thị kết quả
        result_frame = ttk.LabelFrame(main_frame, text="Kết quả nhận diện khuôn mặt & đầu người + Mũ bảo hiểm", padding="10")
        result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Label hiển thị ảnh
        self.image_label = ttk.Label(result_frame, text="Chưa có dữ liệu", 
                                    background="white", anchor="center")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame thống kê mũ bảo hiểm
        helmet_stats_frame = ttk.LabelFrame(main_frame, text="Thống kê Mũ bảo hiểm", padding="10")
        helmet_stats_frame.grid(row=2, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                               padx=(10, 0), pady=(10, 0))
        
        # Thống kê
        ttk.Label(helmet_stats_frame, text="Thống kê:", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(helmet_stats_frame, text="Tổng phát hiện:").grid(row=1, column=0, sticky=tk.W)
        self.total_helmet_label = ttk.Label(helmet_stats_frame, text="0", foreground="blue", font=("Arial", 12, "bold"))
        self.total_helmet_label.grid(row=1, column=1, padx=(10, 0))
        
        ttk.Label(helmet_stats_frame, text="Có mũ bảo hiểm:").grid(row=2, column=0, sticky=tk.W)
        self.with_helmet_label = ttk.Label(helmet_stats_frame, text="0", foreground="green", font=("Arial", 12, "bold"))
        self.with_helmet_label.grid(row=2, column=1, padx=(10, 0))
        
        ttk.Label(helmet_stats_frame, text="Không có mũ:").grid(row=3, column=0, sticky=tk.W)
        self.without_helmet_label = ttk.Label(helmet_stats_frame, text="0", foreground="red", font=("Arial", 12, "bold"))
        self.without_helmet_label.grid(row=3, column=1, padx=(10, 0))
        
        # Button reset thống kê
        ttk.Button(helmet_stats_frame, text="Reset Thống kê", 
                  command=self.reset_helmet_stats).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Frame thông tin
        info_frame = ttk.LabelFrame(main_frame, text="Thông tin", padding="10")
        info_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Text widget cho thông tin
        self.info_text = tk.Text(info_frame, height=8, width=80, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Cấu hình grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
    
    def load_models(self):
        """Load model YOLO, face cascade và helmet model"""
        try:
            self.log_info("Đang tải model YOLO v8...")
            self.model = YOLO('yolov8n.pt')
            self.log_info("✅ Đã tải thành công model YOLO v8!")
            
            self.log_info("Đang tải Haar Cascade...")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.log_info("✅ Đã tải thành công Haar Cascade!")
            
            # Load helmet detection model
            self.log_info("Đang tải model nhận diện mũ bảo hiểm...")
            
            # Tìm model ở các vị trí có thể
            possible_paths = [
                "helmet_detection_project/models/helmet_detection_model.h5",  # Trong dự án helmet
                "models/helmet_detection_model.h5",  # Trong thư mục chính
                "helmet_detection_project/models/best_helmet_model.h5"  # Model tốt nhất
            ]
            
            helmet_model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    helmet_model_path = path
                    break
            
            if helmet_model_path:
                self.helmet_model = tf.keras.models.load_model(helmet_model_path)
                self.log_info(f"✅ Đã tải thành công model từ: {helmet_model_path}")
            else:
                self.log_info("⚠️ Model nhận diện mũ bảo hiểm chưa tồn tại!")
                self.log_info("Hãy train model trước:")
                self.log_info("cd helmet_detection_project")
                self.log_info("python training/train_model.py --mode quick --epochs 20")
                self.helmet_model = None
            
        except Exception as e:
            self.log_info(f"❌ Lỗi khi tải model: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể tải model: {str(e)}")
    
    def predict_helmet(self, face_image):
        """
        Dự đoán mũ bảo hiểm cho một ảnh khuôn mặt
        
        Args:
            face_image: Ảnh khuôn mặt (BGR)
            
        Returns:
            tuple: (prediction, confidence)
        """
        if self.helmet_model is None:
            return "Model chưa load", 0.0
        
        try:
            # Preprocess ảnh
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Dự đoán
            prediction = self.helmet_model.predict(np.expand_dims(face_normalized, axis=0), verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            class_names = ["Không có mũ", "Có mũ bảo hiểm"]
            
            # Logic cân bằng hơn
            if predicted_class == 1:  # Model dự đoán có mũ
                if confidence > 0.5:  # Confidence trung bình
                    return class_names[1], confidence
                else:
                    return class_names[0], confidence  # Không chắc chắn thì không có mũ
            else:  # Model dự đoán không có mũ
                if confidence > 0.5:  # Confidence trung bình
                    return class_names[0], confidence
                else:
                    return class_names[1], confidence  # Không chắc chắn thì có mũ
            
            return class_names[predicted_class], confidence
            
        except Exception as e:
            self.log_info(f"Lỗi khi dự đoán mũ bảo hiểm: {str(e)}")
            return "Lỗi", 0.0
    
    def detect_faces_and_heads_with_helmet(self, frame):
        """Kết hợp phát hiện khuôn mặt, đầu người và nhận diện mũ bảo hiểm"""
        if self.model is None or self.face_cascade is None:
            return frame, 0, []
        
        detected_regions = []
        
        # 1. Phát hiện khuôn mặt bằng Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            aspect_ratio = w / h
            if 0.7 <= aspect_ratio <= 1.4:
                face_roi = gray[y:y+h, x:x+w]
                if face_roi.size > 0:
                    mean_intensity = np.mean(face_roi)
                    std_intensity = np.std(face_roi)
                    if std_intensity > 20 and 50 < mean_intensity < 200:
                        valid_faces.append((x, y, w, h))
        
        # Vẽ bounding box cho khuôn mặt hợp lệ và nhận diện mũ bảo hiểm
        for (x, y, w, h) in valid_faces:
            # Mở rộng để bao gồm cả đầu
            margin_x = int(w * 0.3)
            margin_y = int(h * 0.4)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)
            
            # Cắt vùng khuôn mặt để nhận diện mũ bảo hiểm
            face_roi = frame[y:y+h, x:x+w]
            helmet_prediction, helmet_confidence = self.predict_helmet(face_roi)
            
            # Chọn màu và text dựa trên kết quả nhận diện mũ bảo hiểm
            if "Có mũ bảo hiểm" in helmet_prediction:
                color = (0, 255, 0)  # Xanh lá
                text = "YES"
                self.helmet_stats['with_helmet'] += 1
            else:
                color = (0, 0, 255)  # Đỏ
                text = "NO"
                self.helmet_stats['without_helmet'] += 1
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Vẽ text "YES" hoặc "NO"
            cv2.putText(frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            # Vẽ confidence score
            cv2.putText(frame, f"Conf: {helmet_confidence:.2f}", (x1, y2+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            detected_regions.append((x1, y1, x2-x1, y2-y1))
            self.helmet_stats['total_detections'] += 1
        
        # 2. Phát hiện đầu người bằng YOLO (nếu không có khuôn mặt hợp lệ)
        if len(valid_faces) == 0:
            results = self.model(frame, classes=[0], conf=0.5)
            boxes = results[0].boxes
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    height = y2 - y1
                    width = x2 - x1
                    
                    if height > 100 and width > 50:
                        head_height = height * 0.5
                        
                        head_x1 = max(0, x1 + width * 0.05)
                        head_x2 = min(frame.shape[1], x2 - width * 0.05)
                        head_y1 = max(0, y1 + height * 0.02)
                        head_y2 = min(frame.shape[0], y1 + head_height)
                        
                        # Cắt vùng đầu để nhận diện mũ bảo hiểm
                        head_roi = frame[int(head_y1):int(head_y2), int(head_x1):int(head_x2)]
                        if head_roi.size > 0:
                            helmet_prediction, helmet_confidence = self.predict_helmet(head_roi)
                            
                            # Chọn màu và text
                            if "Có mũ bảo hiểm" in helmet_prediction:
                                color = (0, 255, 0)  # Xanh lá
                                text = "YES"
                                self.helmet_stats['with_helmet'] += 1
                            else:
                                color = (0, 0, 255)  # Đỏ
                                text = "NO"
                                self.helmet_stats['without_helmet'] += 1
                            
                            # Vẽ bounding box
                            cv2.rectangle(frame, (int(head_x1), int(head_y1)), 
                                        (int(head_x2), int(head_y2)), color, 3)
                            
                            # Vẽ text "YES" hoặc "NO"
                            cv2.putText(frame, text, (int(head_x1), int(head_y1)-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                            
                            # Vẽ confidence score
                            cv2.putText(frame, f"Conf: {helmet_confidence:.2f}", (int(head_x1), int(head_y2)+25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            detected_regions.append((int(head_x1), int(head_y1), 
                                                   int(head_x2 - head_x1), int(head_y2 - head_y1)))
                            self.helmet_stats['total_detections'] += 1
        
        return frame, len(detected_regions), detected_regions
    
    def update_helmet_stats(self):
        """Cập nhật thống kê mũ bảo hiểm"""
        self.total_helmet_label.config(text=str(self.helmet_stats['total_detections']))
        self.with_helmet_label.config(text=str(self.helmet_stats['with_helmet']))
        self.without_helmet_label.config(text=str(self.helmet_stats['without_helmet']))
    
    def reset_helmet_stats(self):
        """Reset thống kê mũ bảo hiểm"""
        self.helmet_stats = {
            'total_detections': 0,
            'with_helmet': 0,
            'without_helmet': 0
        }
        self.update_helmet_stats()
        self.log_info("Đã reset thống kê mũ bảo hiểm")
    
    def change_save_folder(self):
        """Thay đổi thư mục lưu ảnh"""
        folder = filedialog.askdirectory(title="Chọn thư mục lưu ảnh")
        if folder:
            self.save_folder = folder
            self.folder_label.config(text=folder)
            self.log_info(f"Đã thay đổi thư mục lưu: {folder}")
    
    def reset_captured_history(self):
        """Reset lịch sử khuôn mặt đã chụp"""
        self.captured_faces_history = []
        self.detection_start_time = 0
        self.log_info("Đã reset lịch sử chụp ảnh")
    
    def should_capture(self, detection_count, regions):
        """Kiểm tra có nên chụp ảnh không - chụp mỗi 2 giây khi có khuôn mặt"""
        current_time = time.time()
        
        if detection_count > 0:
            if current_time - self.last_capture_time >= self.stability_wait_time:
                return True, "Chụp định kỳ mỗi 2 giây"
        
        return False, ""
    
    def save_head_image(self, frame, head_position):
        """Lưu ảnh đầu người với cải tiến"""
        x, y, w, h = head_position
        
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.25)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + w + margin_x)
        y2 = min(frame.shape[0], y + h + margin_y)
        
        if (x2 - x1) < 150 or (y2 - y1) < 150:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            size = max(150, max(x2 - x1, y2 - y1))
            x1 = max(0, center_x - size // 2)
            y1 = max(0, center_y - size // 2)
            x2 = min(frame.shape[1], center_x + size // 2)
            y2 = min(frame.shape[0], center_y + size // 2)
        
        head_img = frame[y1:y2, x1:x2]
        
        if head_img.size == 0:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"head_{timestamp}.jpg"
        filepath = os.path.join(self.save_folder, filename)
        
        success = cv2.imwrite(filepath, head_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            return filepath
        else:
            return None
    
    def log_info(self, message):
        """Ghi thông tin vào text widget"""
        self.info_text.insert(tk.END, f"{message}\n")
        self.info_text.see(tk.END)
        self.root.update()
    
    def start_camera_detection(self):
        """Bắt đầu nhận diện từ camera"""
        if self.model is None:
            messagebox.showerror("Lỗi", "Model YOLO chưa được tải!")
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera!")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.camera_running = True
        self.camera_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        self.log_info("Đã bắt đầu nhận diện đầu người + mũ bảo hiểm từ camera...")
        self.log_info(f"Thư mục lưu ảnh: {self.save_folder}")
        
        thread = threading.Thread(target=self.camera_detection_loop)
        thread.daemon = True
        thread.start()
    
    def camera_detection_loop(self):
        """Vòng lặp nhận diện camera với logic chụp mới"""
        try:
            while self.camera_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.log_info("❌ Không thể đọc frame từ camera - Thử lại...")
                    time.sleep(0.1)
                    continue
                
                try:
                    annotated_frame, detection_count, regions = self.detect_faces_and_heads_with_helmet(frame.copy())
                except Exception as e:
                    self.log_info(f"❌ Lỗi trong detection: {str(e)}")
                    continue
                
                current_time = time.time()
                
                should_capture, reason = self.should_capture(detection_count, regions)
                
                if should_capture and len(regions) > 0:
                    region_pos = regions[0]
                    filepath = self.save_head_image(frame, region_pos)
                    if filepath:
                        self.log_info(f"✅ {reason}: {os.path.basename(filepath)}")
                        self.last_capture_time = current_time
                    else:
                        self.log_info("❌ Không thể lưu ảnh")
                
                self.last_detection_count = detection_count
                
                # Cập nhật thống kê mũ bảo hiểm
                self.update_helmet_stats()
                
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                height, width = rgb_frame.shape[:2]
                max_width = 600
                if width > max_width:
                    scale = max_width / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
                
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
                if detection_count > 0:
                    if self.last_capture_time > 0:
                        remaining_time = max(0, self.stability_wait_time - (current_time - self.last_capture_time))
                        self.log_info(f"Phát hiện {detection_count} khuôn mặt/đầu - Chụp tiếp sau {remaining_time:.1f}s")
                    else:
                        self.log_info(f"Phát hiện {detection_count} khuôn mặt/đầu - Sẵn sàng chụp")
                else:
                    if current_time - self.last_no_detection_log > 3.0:
                        self.log_info("Không phát hiện khuôn mặt/đầu người nào")
                        self.last_no_detection_log = current_time
        
        except Exception as e:
            self.log_info(f"❌ Lỗi trong camera loop: {str(e)}")
        finally:
            self.cap.release()
            self.log_info("Camera đã được giải phóng")
    
    def stop_camera(self):
        """Dừng camera"""
        self.camera_running = False
        self.camera_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log_info("Đã dừng camera")
    
    def detect_from_image(self):
        """Nhận diện từ ảnh"""
        if self.model is None:
            messagebox.showerror("Lỗi", "Model YOLO chưa được tải!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if not file_path:
            return
        
        try:
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Lỗi", "Không thể đọc file ảnh!")
                return
            
            annotated_image, detection_count, regions = self.detect_faces_and_heads_with_helmet(image.copy())
            
            for i, region_pos in enumerate(regions):
                filepath = self.save_head_image(image, region_pos)
                if filepath:
                    self.log_info(f"Đã lưu vùng {i+1}: {os.path.basename(filepath)}")
                else:
                    self.log_info(f"Không thể lưu vùng {i+1}")
            
            rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            height, width = rgb_image.shape[:2]
            max_width = 800
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                rgb_image = cv2.resize(rgb_image, (new_width, new_height))
            
            pil_image = Image.fromarray(rgb_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            if detection_count > 0:
                self.log_info(f"Phát hiện {detection_count} khuôn mặt/đầu trong ảnh")
            else:
                self.log_info("Không phát hiện khuôn mặt/đầu nào trong ảnh")
            
            output_path = f"head_result_{os.path.basename(file_path)}"
            cv2.imwrite(output_path, annotated_image)
            self.log_info(f"Đã lưu kết quả: {output_path}")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi xử lý ảnh: {str(e)}")
            self.log_info(f"Lỗi: {str(e)}")
    
    def detect_from_video(self):
        """Nhận diện từ video"""
        if self.model is None:
            messagebox.showerror("Lỗi", "Model YOLO chưa được tải!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        
        if not file_path:
            return
        
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở file video!")
                return
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.log_info(f"Đang xử lý video: {os.path.basename(file_path)}")
            self.log_info(f"Kích thước: {width}x{height}, FPS: {fps}")
            
            output_path = f"head_result_{os.path.basename(file_path)}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            captured_heads = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                annotated_frame, detection_count, regions = self.detect_faces_and_heads_with_helmet(frame.copy())
                
                for region_pos in regions:
                    filepath = self.save_head_image(frame, region_pos)
                    if filepath:
                        captured_heads += 1
                        self.log_info(f"Đã chụp từ video: {os.path.basename(filepath)}")
                
                self.last_detection_count = detection_count
                out.write(annotated_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.log_info(f"Tiến độ: {progress:.1f}% ({frame_count}/{total_frames}) - Đã chụp {captured_heads} vùng")
            
            cap.release()
            out.release()
            
            self.log_info(f"Hoàn thành xử lý video! Đã lưu: {output_path}")
            self.log_info(f"Tổng cộng đã chụp {captured_heads} vùng từ video")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi xử lý video: {str(e)}")
            self.log_info(f"Lỗi: {str(e)}")


def main():
    root = tk.Tk()
    app = HeadDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
