"""
Ứng dụng Nhận diện Khuôn mặt với tính năng nhận diện mũ bảo hiểm
- Nhận diện từ ảnh
- Nhận diện từ camera trực tiếp (chụp ảnh mỗi 2 giây)
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

# Import TensorFlow với xử lý lỗi
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None


class HeadDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Nhận diện Khuôn mặt + Mũ bảo hiểm")
        self.root.geometry("900x650")
        
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
        
        # Tạo thư mục con để phân loại ảnh có mũ và không có mũ
        self.with_helmet_folder = os.path.join(self.save_folder, "with_helmet")
        self.without_helmet_folder = os.path.join(self.save_folder, "without_helmet")
        if not os.path.exists(self.with_helmet_folder):
            os.makedirs(self.with_helmet_folder)
        if not os.path.exists(self.without_helmet_folder):
            os.makedirs(self.without_helmet_folder)
        
        # Biến để theo dõi chụp ảnh
        self.last_capture_time = 0
        self.capture_interval = 2.0  # Chụp mỗi 2 giây
        
        # Queue để xử lý inference song song (tối ưu tốc độ)
        self.inference_queue = []
        self.inference_lock = threading.Lock()
        
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
        title_label = ttk.Label(main_frame, text="Ứng dụng Nhận diện Khuôn mặt + Mũ bảo hiểm", 
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
        
        self.stop_btn = ttk.Button(mode_frame, text="Dừng Camera", 
                                  command=self.stop_camera, state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=5)
        
        # Frame cài đặt
        settings_frame = ttk.LabelFrame(main_frame, text="Cài đặt", padding="10")
        settings_frame.grid(row=1, column=2, sticky=(tk.W, tk.E), padx=(10, 0))
        
        ttk.Label(settings_frame, text="Thư mục lưu:").grid(row=0, column=0, padx=5)
        self.folder_label = ttk.Label(settings_frame, text=self.save_folder, foreground="blue")
        self.folder_label.grid(row=0, column=1, padx=5)
        
        ttk.Button(settings_frame, text="Đổi thư mục", 
                  command=self.change_save_folder).grid(row=0, column=2, padx=5)
        
        # Frame hiển thị kết quả
        result_frame = ttk.LabelFrame(main_frame, text="Kết quả nhận diện", padding="10")
        result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Label hiển thị ảnh
        self.image_label = ttk.Label(result_frame, text="Chưa có dữ liệu", 
                                    background="white", anchor="center")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame thông tin
        info_frame = ttk.LabelFrame(main_frame, text="Thông tin", padding="10")
        info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
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
            
            # Load helmet detection model từ apps/model
            self.log_info("Đang tải model nhận diện mũ bảo hiểm từ apps/model...")
            possible_paths = [
                "model/best_model.h5",  # Từ cùng thư mục apps
                "apps/model/best_model.h5",  # Từ root
                "../apps/model/best_model.h5",  # Từ app/
                "model/final_model.h5",  # Backup từ cùng thư mục apps
                "apps/model/final_model.h5",  # Backup từ root
                "../apps/model/final_model.h5"  # Backup từ app/
            ]
            
            helmet_model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    helmet_model_path = path
                    break
            
            if helmet_model_path:
                if not TENSORFLOW_AVAILABLE:
                    self.log_info("❌ TensorFlow chưa được cài đặt!")
                    self.log_info("   Hãy chạy: pip install tensorflow")
                    self.helmet_model = None
                else:
                    try:
                        self.helmet_model = keras.models.load_model(helmet_model_path)
                        self.log_info(f"✅ Đã tải thành công model từ: {helmet_model_path}")
                    except Exception as e:
                        self.log_info(f"❌ Lỗi khi load model: {str(e)}")
                        self.helmet_model = None
            else:
                self.log_info("⚠️ Model nhận diện mũ bảo hiểm chưa tồn tại!")
                self.log_info("   Tìm trong: model/best_model.h5 hoặc apps/model/best_model.h5")
                self.helmet_model = None
            
        except Exception as e:
            self.log_info(f"❌ Lỗi khi tải model: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể tải model: {str(e)}")
    
    def predict_helmet(self, face_image):
        """
        Dự đoán mũ bảo hiểm cho một ảnh khuôn mặt (tối ưu tốc độ)
        
        Args:
            face_image: Ảnh khuôn mặt (BGR numpy array)
            
        Returns:
            tuple: (has_helmet: bool, confidence: float, class_name: str)
        """
        if self.helmet_model is None:
            return False, 0.0, "Chưa có model"
        
        try:
            # Convert BGR to RGB
            if len(face_image.shape) == 2:
                # Nếu là grayscale, chuyển sang BGR rồi RGB
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            else:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Resize về kích thước model yêu cầu (224x224)
            face_resized = cv2.resize(face_rgb, (224, 224))
            
            # Normalize [0, 255] -> [0, 1]
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Predict (batch size=1 để tối ưu)
            prediction = self.helmet_model.predict(
                np.expand_dims(face_normalized, axis=0), 
                verbose=0
            )
            
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            # Model từ home_work_4: class 0 = no_helmet, class 1 = with_helmet
            has_helmet = (predicted_class_idx == 1)
            class_name = "Có mũ bảo hiểm" if has_helmet else "Không có mũ"
            
            return has_helmet, confidence, class_name
            
        except Exception as e:
            self.log_info(f"Lỗi khi dự đoán mũ bảo hiểm: {str(e)}")
            import traceback
            self.log_info(traceback.format_exc())
            return False, 0.0, "Lỗi"
    
    def detect_faces(self, frame):
        """
        Phát hiện khuôn mặt và đầu người trong frame - Chỉ chọn 1 khuôn mặt/đầu gần nhất
        - Haar Cascade: Phát hiện khuôn mặt (chạy song song)
        - YOLO: Phát hiện đầu người (chạy song song, hoạt động tốt kể cả khi đeo khẩu trang)
        - Chọn khuôn mặt/đầu lớn nhất (gần nhất) để xử lý
        """
        detected_regions = []
        
        # Danh sách để lưu tất cả các detection (face và head)
        all_detections = []  # Format: (x1, y1, x2, y2, area, type, face_roi_or_head_roi, face_box_or_head_box)
        # type: 'face' hoặc 'head'
        # face_roi_or_head_roi: ROI để predict helmet
        # face_box_or_head_box: (x, y, w, h) cho face hoặc (head_x1, head_y1, head_x2, head_y2) cho head
        
        # 1. Phát hiện khuôn mặt bằng Haar Cascade (chạy song song)
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,  # Giảm để nhận diện tốt hơn với khẩu trang
                minSize=(60, 60),
                maxSize=(300, 300),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Lọc và lưu các khuôn mặt hợp lệ
            for (x, y, w, h) in faces:
                aspect_ratio = w / h
                if 0.6 <= aspect_ratio <= 1.5:
                    face_roi_gray = gray[y:y+h, x:x+w]
                    if face_roi_gray.size > 0:
                        mean_intensity = np.mean(face_roi_gray)
                        std_intensity = np.std(face_roi_gray)
                        if std_intensity > 15 and 40 < mean_intensity < 220:
                            # Mở rộng để bao gồm cả đầu
                            margin_x = int(w * 0.3)
                            margin_y = int(h * 0.4)
                            x1 = max(0, x - margin_x)
                            y1 = max(0, y - margin_y)
                            x2 = min(frame.shape[1], x + w + margin_x)
                            y2 = min(frame.shape[0], y + h + margin_y)
                            
                            area = (x2 - x1) * (y2 - y1)
                            face_roi_color = frame[y:y+h, x:x+w]  # ROI để predict
                            all_detections.append((x1, y1, x2, y2, area, 'face', face_roi_color, (x, y, w, h)))
        
        # 2. Phát hiện đầu người bằng YOLO (chạy song song)
        if self.model is not None:
            try:
                results = self.model(frame, classes=[0], conf=0.5)  # Class 0 = person
                boxes = results[0].boxes
                
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        height = y2 - y1
                        width = x2 - x1
                        
                        # Chỉ xử lý nếu kích thước hợp lý
                        if height > 100 and width > 50:
                            # Tính vùng đầu (top 45% của người)
                            head_height = height * 0.45
                            
                            head_x1 = max(0, int(x1 + width * 0.1))
                            head_x2 = min(frame.shape[1], int(x2 - width * 0.1))
                            head_y1 = max(0, int(y1 + height * 0.05))
                            head_y2 = min(frame.shape[0], int(y1 + head_height))
                            
                            head_roi = frame[head_y1:head_y2, head_x1:head_x2]
                            if head_roi.size > 0:
                                area = (head_x2 - head_x1) * (head_y2 - head_y1)
                                all_detections.append((head_x1, head_y1, head_x2, head_y2, area, 'head', head_roi, (head_x1, head_y1, head_x2, head_y2)))
            except Exception as e:
                # Lỗi khi sử dụng YOLO, bỏ qua
                pass
        
        # 3. Chọn khuôn mặt/đầu gần nhất (lớn nhất)
        if len(all_detections) > 0:
            # Sắp xếp theo diện tích giảm dần (lớn nhất = gần nhất)
            all_detections.sort(key=lambda d: d[4], reverse=True)
            
            # Chọn detection lớn nhất
            best_detection = all_detections[0]
            x1, y1, x2, y2, area, det_type, roi, box_info = best_detection
            
            # Nhận diện mũ bảo hiểm
            has_helmet = False
            helmet_confidence = 0.0
            color = (255, 0, 0)  # Mặc định xanh dương
            
            if self.helmet_model is not None and roi.size > 0:
                has_helmet, helmet_confidence, class_name = self.predict_helmet(roi)
                
                # Chọn màu và text
                if has_helmet:
                    color = (0, 255, 0)  # Xanh lá - Có mũ
                    text = "YES"
                else:
                    color = (0, 0, 255)  # Đỏ - Không có mũ
                    text = "NO"
                
                # Vẽ text "YES" hoặc "NO"
                cv2.putText(frame, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                
                # Vẽ confidence score
                cv2.putText(frame, f"Conf: {helmet_confidence:.2f}", (x1, y2+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Vẽ label để phân biệt phương pháp phát hiện
                method_label = "Haar" if det_type == 'face' else "YOLO"
                cv2.putText(frame, method_label, (x1, y2+50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                # Vẽ label khi không có model mũ
                method_label = "Haar" if det_type == 'face' else "YOLO"
                cv2.putText(frame, method_label, (x1, y2+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Lưu thông tin
            detected_regions.append((x1, y1, x2-x1, y2-y1, has_helmet, helmet_confidence))
        
        return frame, len(detected_regions), detected_regions
    
    def change_save_folder(self):
        """Thay đổi thư mục lưu ảnh"""
        folder = filedialog.askdirectory(title="Chọn thư mục lưu ảnh")
        if folder:
            self.save_folder = folder
            self.folder_label.config(text=folder)
            
            # Tạo lại thư mục con cho có mũ và không có mũ
            self.with_helmet_folder = os.path.join(self.save_folder, "with_helmet")
            self.without_helmet_folder = os.path.join(self.save_folder, "without_helmet")
            if not os.path.exists(self.with_helmet_folder):
                os.makedirs(self.with_helmet_folder)
            if not os.path.exists(self.without_helmet_folder):
                os.makedirs(self.without_helmet_folder)
            
            self.log_info(f"Đã thay đổi thư mục lưu: {folder}")
            self.log_info(f"  - Có mũ: {self.with_helmet_folder}")
            self.log_info(f"  - Không có mũ: {self.without_helmet_folder}")
    
    def save_face_image(self, frame, face_position, has_helmet=None):
        """
        Lưu ảnh khuôn mặt và phân loại vào folder tương ứng
        
        Args:
            frame: Frame ảnh gốc
            face_position: (x, y, w, h) hoặc (x, y, w, h, has_helmet, confidence)
            has_helmet: None (tự động xác định từ face_position) hoặc bool
            
        Returns:
            filepath: Đường dẫn file đã lưu hoặc None
        """
        # Xử lý face_position có thể có thêm thông tin
        if len(face_position) >= 5:
            x, y, w, h = face_position[0:4]
            if has_helmet is None and len(face_position) >= 5:
                has_helmet = face_position[4]
        else:
            x, y, w, h = face_position
        
        # Cắt vùng khuôn mặt
        face_img = frame[y:y+h, x:x+w]
        
        if face_img.size == 0:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Chọn thư mục dựa trên kết quả nhận diện mũ
        if has_helmet is True:
            # Có mũ bảo hiểm - lưu vào thư mục with_helmet
            filename = f"face_with_helmet_{timestamp}.jpg"
            filepath = os.path.join(self.with_helmet_folder, filename)
        elif has_helmet is False:
            # Không có mũ bảo hiểm - lưu vào thư mục without_helmet
            filename = f"face_no_helmet_{timestamp}.jpg"
            filepath = os.path.join(self.without_helmet_folder, filename)
        else:
            # Không xác định được - lưu vào thư mục chính
            filename = f"face_{timestamp}.jpg"
            filepath = os.path.join(self.save_folder, filename)
        
        success = cv2.imwrite(filepath, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
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
        if self.face_cascade is None:
            messagebox.showerror("Lỗi", "Model Haar Cascade chưa được tải!")
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
        
        self.log_info("Đã bắt đầu nhận diện khuôn mặt từ camera...")
        self.log_info(f"Thư mục lưu ảnh: {self.save_folder}")
        self.log_info("⚠️ Ảnh sẽ được chụp tự động mỗi 2 giây khi phát hiện khuôn mặt")
        
        thread = threading.Thread(target=self.camera_detection_loop)
        thread.daemon = True
        thread.start()
    
    def camera_detection_loop(self):
        """Vòng lặp nhận diện camera - chụp ảnh mỗi 2 giây"""
        try:
            while self.camera_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.log_info("❌ Không thể đọc frame từ camera - Thử lại...")
                    time.sleep(0.1)
                    continue
                
                try:
                    annotated_frame, detection_count, regions = self.detect_faces(frame.copy())
                except Exception as e:
                    self.log_info(f"❌ Lỗi trong detection: {str(e)}")
                    continue
                
                current_time = time.time()
                
                # Chụp ảnh mỗi 2 giây khi có khuôn mặt
                if detection_count > 0 and len(regions) > 0:
                    if current_time - self.last_capture_time >= self.capture_interval:
                        # Chụp tất cả khuôn mặt phát hiện được (đã có kết quả nhận diện mũ)
                        for region_pos in regions:
                            filepath = self.save_face_image(frame, region_pos)
                            if filepath:
                                # Xác định loại từ region_pos
                                has_helmet_info = ""
                                if len(region_pos) >= 5:
                                    has_helmet = region_pos[4]
                                    confidence = region_pos[5] if len(region_pos) >= 6 else 0.0
                                    helmet_type = "Có mũ" if has_helmet else "Không có mũ"
                                    has_helmet_info = f" - {helmet_type} ({confidence:.2f})"
                                
                                self.log_info(f"✅ Đã chụp{has_helmet_info}: {os.path.basename(filepath)}")
                            else:
                                self.log_info("❌ Không thể lưu ảnh")
                        self.last_capture_time = current_time
                
                # Hiển thị frame
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
                
                # Hiển thị thông tin
                if detection_count > 0:
                    remaining_time = max(0, self.capture_interval - (current_time - self.last_capture_time))
                    self.log_info(f"Phát hiện {detection_count} khuôn mặt - Chụp tiếp sau {remaining_time:.1f}s")
                else:
                    self.log_info("Không phát hiện khuôn mặt nào")
                
                time.sleep(0.1)  # Giảm tải CPU
        
        except Exception as e:
            self.log_info(f"❌ Lỗi trong camera loop: {str(e)}")
        finally:
            if self.cap is not None:
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
        if self.face_cascade is None:
            messagebox.showerror("Lỗi", "Model Haar Cascade chưa được tải!")
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
            
            annotated_image, detection_count, regions = self.detect_faces(image.copy())
            
            # Lưu tất cả khuôn mặt phát hiện được (đã có kết quả nhận diện mũ)
            for i, region_pos in enumerate(regions):
                filepath = self.save_face_image(image, region_pos)
                if filepath:
                    # Xác định loại từ region_pos
                    has_helmet_info = ""
                    if len(region_pos) >= 5:
                        has_helmet = region_pos[4]
                        confidence = region_pos[5] if len(region_pos) >= 6 else 0.0
                        helmet_type = "Có mũ" if has_helmet else "Không có mũ"
                        has_helmet_info = f" - {helmet_type} ({confidence:.2f})"
                    
                    self.log_info(f"Đã lưu vùng {i+1}{has_helmet_info}: {os.path.basename(filepath)}")
                else:
                    self.log_info(f"Không thể lưu vùng {i+1}")
            
            # Hiển thị ảnh
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
                self.log_info(f"Phát hiện {detection_count} khuôn mặt trong ảnh")
            else:
                self.log_info("Không phát hiện khuôn mặt nào trong ảnh")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi xử lý ảnh: {str(e)}")
            self.log_info(f"Lỗi: {str(e)}")


def main():
    root = tk.Tk()
    app = HeadDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
