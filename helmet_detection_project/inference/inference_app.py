"""
Ứng dụng inference để nhận diện mũ bảo hiểm
Sử dụng model đã train để dự đoán real-time
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time
from datetime import datetime
import tensorflow as tf
from models.helmet_model import HelmetDetectionModel


class HelmetInferenceApp:
    """Ứng dụng inference nhận diện mũ bảo hiểm"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện Mũ bảo hiểm - Inference App")
        self.root.geometry("1000x700")
        
        # Khởi tạo camera
        self.cap = None
        self.camera_running = False
        
        # Load model
        self.model = None
        self.model_loaded = False
        self.load_model()
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Biến thống kê
        self.total_detections = 0
        self.helmet_detections = 0
        self.no_helmet_detections = 0
        
        # Tạo giao diện
        self.create_widgets()
        
        # Log thông tin khởi động
        self.log_info("Ứng dụng inference đã khởi động")
        if self.model_loaded:
            self.log_info("✅ Model đã được load thành công")
        else:
            self.log_info("❌ Không thể load model. Hãy train model trước.")
    
    def load_model(self):
        """Load model đã train"""
        try:
            model_path = "models/helmet_detection_model.h5"
            if os.path.exists(model_path):
                self.model = HelmetDetectionModel()
                self.model.load_model(model_path)
                self.model_loaded = True
                print("✅ Model loaded successfully")
            else:
                print("❌ Model file not found")
                self.model_loaded = False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model_loaded = False
    
    def create_widgets(self):
        """Tạo giao diện"""
        # Frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tiêu đề
        title_label = ttk.Label(main_frame, text="Nhận diện Mũ bảo hiểm - Inference", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Frame điều khiển camera
        control_frame = ttk.LabelFrame(main_frame, text="Điều khiển Camera", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_btn = ttk.Button(control_frame, text="Bắt đầu Camera", 
                                   command=self.start_camera)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Dừng Camera", 
                                  command=self.stop_camera, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        # Button test ảnh
        self.test_image_btn = ttk.Button(control_frame, text="Test từ Ảnh", 
                                        command=self.test_from_image)
        self.test_image_btn.grid(row=0, column=2, padx=5)
        
        # Frame hiển thị ảnh
        image_frame = ttk.LabelFrame(main_frame, text="Camera Feed & Kết quả", padding="10")
        image_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="Chưa có dữ liệu", 
                                    background="white", anchor="center")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame thống kê và kết quả
        stats_frame = ttk.LabelFrame(main_frame, text="Thống kê & Kết quả", padding="10")
        stats_frame.grid(row=2, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                        padx=(10, 0), pady=(0, 10))
        
        # Thống kê
        ttk.Label(stats_frame, text="Thống kê:", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(stats_frame, text="Tổng phát hiện:").grid(row=1, column=0, sticky=tk.W)
        self.total_label = ttk.Label(stats_frame, text="0", foreground="blue", font=("Arial", 12, "bold"))
        self.total_label.grid(row=1, column=1, padx=(10, 0))
        
        ttk.Label(stats_frame, text="Có mũ bảo hiểm:").grid(row=2, column=0, sticky=tk.W)
        self.helmet_label = ttk.Label(stats_frame, text="0", foreground="green", font=("Arial", 12, "bold"))
        self.helmet_label.grid(row=2, column=1, padx=(10, 0))
        
        ttk.Label(stats_frame, text="Không có mũ:").grid(row=3, column=0, sticky=tk.W)
        self.no_helmet_label = ttk.Label(stats_frame, text="0", foreground="red", font=("Arial", 12, "bold"))
        self.no_helmet_label.grid(row=3, column=1, padx=(10, 0))
        
        # Kết quả hiện tại
        ttk.Label(stats_frame, text="Kết quả hiện tại:", font=("Arial", 12, "bold")).grid(row=4, column=0, columnspan=2, pady=(20, 10))
        
        self.current_result_label = ttk.Label(stats_frame, text="Chưa có kết quả", 
                                            font=("Arial", 14, "bold"), foreground="gray")
        self.current_result_label.grid(row=5, column=0, columnspan=2, pady=5)
        
        self.confidence_label = ttk.Label(stats_frame, text="", font=("Arial", 10))
        self.confidence_label.grid(row=6, column=0, columnspan=2, pady=5)
        
        # Button reset thống kê
        ttk.Button(stats_frame, text="Reset Thống kê", 
                  command=self.reset_statistics).grid(row=7, column=0, columnspan=2, pady=10)
        
        # Frame thông tin
        info_frame = ttk.LabelFrame(main_frame, text="Thông tin", padding="10")
        info_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
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
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
    
    def log_info(self, message):
        """Ghi thông tin"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.info_text.see(tk.END)
        self.root.update()
    
    def start_camera(self):
        """Bắt đầu camera"""
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa được load! Hãy train model trước.")
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera!")
            return
        
        # Cài đặt camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.camera_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        self.log_info("Đã bắt đầu camera")
        
        # Chạy camera loop
        thread = threading.Thread(target=self.camera_loop)
        thread.daemon = True
        thread.start()
    
    def camera_loop(self):
        """Vòng lặp camera với inference"""
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Phát hiện khuôn mặt
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            
            # Xử lý từng khuôn mặt
            for (x, y, w, h) in faces:
                # Cắt vùng khuôn mặt
                face_roi = frame[y:y+h, x:x+w]
                
                # Dự đoán mũ bảo hiểm
                prediction, confidence = self.predict_helmet(face_roi)
                
                # Vẽ bounding box với màu tương ứng
                if prediction == "Có mũ bảo hiểm":
                    color = (0, 255, 0)  # Xanh lá
                    self.helmet_detections += 1
                else:
                    color = (0, 0, 255)  # Đỏ
                    self.no_helmet_detections += 1
                
                # Vẽ rectangle và text
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{prediction}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Cập nhật thống kê
                self.total_detections += 1
                self.update_statistics()
                
                # Log kết quả
                self.log_info(f"Phát hiện: {prediction} (Confidence: {confidence:.2f})")
            
            # Hiển thị frame
            self.display_frame(frame)
    
    def predict_helmet(self, face_image):
        """
        Dự đoán mũ bảo hiểm cho một ảnh khuôn mặt
        
        Args:
            face_image: Ảnh khuôn mặt (BGR)
            
        Returns:
            tuple: (prediction, confidence)
        """
        try:
            # Preprocess ảnh
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Dự đoán
            prediction, confidence = self.model.predict_single_image(face_normalized)
            
            return prediction, confidence
            
        except Exception as e:
            self.log_info(f"Lỗi khi dự đoán: {str(e)}")
            return "Lỗi", 0.0
    
    def update_statistics(self):
        """Cập nhật thống kê"""
        self.total_label.config(text=str(self.total_detections))
        self.helmet_label.config(text=str(self.helmet_detections))
        self.no_helmet_label.config(text=str(self.no_helmet_detections))
        
        # Cập nhật kết quả hiện tại
        if self.total_detections > 0:
            helmet_ratio = self.helmet_detections / self.total_detections
            if helmet_ratio > 0.5:
                self.current_result_label.config(text="Có mũ bảo hiểm", foreground="green")
            else:
                self.current_result_label.config(text="Không có mũ", foreground="red")
            
            self.confidence_label.config(text=f"Tỷ lệ có mũ: {helmet_ratio:.1%}")
    
    def reset_statistics(self):
        """Reset thống kê"""
        self.total_detections = 0
        self.helmet_detections = 0
        self.no_helmet_detections = 0
        
        self.total_label.config(text="0")
        self.helmet_label.config(text="0")
        self.no_helmet_label.config(text="0")
        self.current_result_label.config(text="Chưa có kết quả", foreground="gray")
        self.confidence_label.config(text="")
        
        self.log_info("Đã reset thống kê")
    
    def test_from_image(self):
        """Test từ ảnh"""
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa được load!")
            return
        
        # Chọn file ảnh
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh để test",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
        
        try:
            # Đọc ảnh
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Lỗi", "Không thể đọc file ảnh!")
                return
            
            # Phát hiện khuôn mặt
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            
            if len(faces) == 0:
                messagebox.showwarning("Cảnh báo", "Không phát hiện khuôn mặt trong ảnh!")
                return
            
            # Xử lý từng khuôn mặt
            for (x, y, w, h) in faces:
                face_roi = image[y:y+h, x:x+w]
                prediction, confidence = self.predict_helmet(face_roi)
                
                # Vẽ kết quả
                if prediction == "Có mũ bảo hiểm":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image, f"{prediction}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(image, f"Conf: {confidence:.2f}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Hiển thị kết quả
            self.display_frame(image)
            self.log_info(f"Đã test ảnh: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi xử lý ảnh: {str(e)}")
            self.log_info(f"Lỗi: {str(e)}")
    
    def display_frame(self, frame):
        """Hiển thị frame"""
        # Resize frame
        height, width = frame.shape[:2]
        max_width = 600
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Chuyển đổi sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Cập nhật giao diện
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def stop_camera(self):
        """Dừng camera"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
        self.log_info("Đã dừng camera")


def main():
    """Hàm main"""
    root = tk.Tk()
    app = HelmetInferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


