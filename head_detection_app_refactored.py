"""
Ứng dụng Nhận diện Khuôn mặt & Đầu người - Phiên bản Refactored
Sử dụng kiến trúc modular để dễ bảo trì và mở rộng
"""

import tkinter as tk
import cv2
import threading
import time
import os
from config import AppConfig
from detection import FaceHeadDetector
from capture import ImageCaptureManager
from gui import AppGUI


class HeadDetectionApp:
    """Class chính của ứng dụng nhận diện khuôn mặt và đầu người"""
    
    def __init__(self, root):
        self.root = root
        
        # Khởi tạo các component
        self.detector = FaceHeadDetector()
        self.capture_manager = ImageCaptureManager(AppConfig.get_session_folder())
        self.gui = AppGUI(root, self)
        
        # Biến trạng thái
        self.camera_running = False
        self.cap = None
        self.last_detection_count = 0
        self.last_no_detection_log = 0
        
        # Thông báo khởi động
        self.gui.log_info(f"📁 Thư mục lưu ảnh: {self.capture_manager.save_folder}")
    
    def start_camera_detection(self):
        """Bắt đầu nhận diện từ camera"""
        if not self.detector.is_model_loaded():
            self.gui.show_error("Lỗi", "Model chưa được tải!")
            return
        
        # Khởi tạo camera với cài đặt tốt hơn
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.gui.show_error("Lỗi", "Không thể mở camera!")
            return
        
        # Cài đặt camera
        self._configure_camera()
        
        self.camera_running = True
        self.gui.set_camera_buttons_state(True)
        
        self.gui.log_info("Đã bắt đầu nhận diện đầu người từ camera...")
        self.gui.log_info(f"Thư mục lưu ảnh: {self.capture_manager.save_folder}")
        
        # Chạy detection trong thread riêng
        thread = threading.Thread(target=self._camera_detection_loop)
        thread.daemon = True
        thread.start()
    
    def _configure_camera(self):
        """Cấu hình camera"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, AppConfig.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, AppConfig.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, AppConfig.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, AppConfig.CAMERA_BUFFER_SIZE)
    
    def _camera_detection_loop(self):
        """Vòng lặp nhận diện camera"""
        try:
            while self.camera_running:
                ret, frame = self.cap.read()
                if not ret:
                    self.gui.log_info("❌ Không thể đọc frame từ camera - Thử lại...")
                    time.sleep(0.1)
                    continue
                
                # Thực hiện detection
                try:
                    annotated_frame, detection_count, regions = self.detector.detect_faces_and_heads(frame.copy())
                except Exception as e:
                    self.gui.log_info(f"❌ Lỗi trong detection: {str(e)}")
                    continue
                
                current_time = time.time()
                
                # Logic chụp ảnh
                should_capture, reason = self.capture_manager.should_capture(detection_count)
                
                if should_capture and len(regions) > 0:
                    region_pos = regions[0]
                    filepath = self.capture_manager.save_head_image(frame, region_pos)
                    if filepath:
                        self.gui.log_info(f"✅ {reason}: {os.path.basename(filepath)}")
                    else:
                        self.gui.log_info("❌ Không thể lưu ảnh")
                
                # Cập nhật trạng thái phát hiện
                self.last_detection_count = detection_count
                
                # Cập nhật giao diện
                self.gui.update_image_display(annotated_frame)
                self._update_detection_info(detection_count, current_time)
        
        except Exception as e:
            self.gui.log_info(f"❌ Lỗi trong camera loop: {str(e)}")
        finally:
            self.cap.release()
            self.gui.log_info("Camera đã được giải phóng")
    
    def _update_detection_info(self, detection_count, current_time):
        """Cập nhật thông tin phát hiện"""
        if detection_count > 0:
            # Hiển thị thời gian còn lại đến lần chụp tiếp theo
            remaining_time = self.capture_manager.get_remaining_time()
            if remaining_time > 0:
                self.gui.log_info(f"Phát hiện {detection_count} khuôn mặt/đầu - Chụp tiếp sau {remaining_time:.1f}s")
            else:
                self.gui.log_info(f"Phát hiện {detection_count} khuôn mặt/đầu - Sẵn sàng chụp")
        else:
            # Thông báo khi không phát hiện gì (chỉ log mỗi 3 giây để tránh spam)
            if current_time - self.last_no_detection_log > AppConfig.NO_DETECTION_LOG_INTERVAL:
                self.gui.log_info("Không phát hiện khuôn mặt/đầu người nào")
                self.last_no_detection_log = current_time
    
    def stop_camera(self):
        """Dừng camera"""
        self.camera_running = False
        self.gui.set_camera_buttons_state(False)
        self.gui.log_info("Đã dừng camera")
    
    def detect_from_image(self):
        """Nhận diện từ ảnh"""
        if not self.detector.is_model_loaded():
            self.gui.show_error("Lỗi", "Model chưa được tải!")
            return
        
        # Chọn file ảnh
        file_path = self.gui.ask_file_path(
            "Chọn ảnh", 
            AppConfig.get_file_filters()['images']
        )
        
        if not file_path:
            return
        
        try:
            # Đọc ảnh
            image = cv2.imread(file_path)
            if image is None:
                self.gui.show_error("Lỗi", "Không thể đọc file ảnh!")
                return
            
            # Thực hiện detection
            annotated_image, detection_count, regions = self.detector.detect_faces_and_heads(image.copy())
            
            # Chụp và lưu tất cả vùng phát hiện
            for i, region_pos in enumerate(regions):
                filepath = self.capture_manager.save_head_image(image, region_pos)
                if filepath:
                    self.gui.log_info(f"Đã lưu vùng {i+1}: {os.path.basename(filepath)}")
                else:
                    self.gui.log_info(f"Không thể lưu vùng {i+1}")
            
            # Cập nhật giao diện
            self.gui.update_image_display(annotated_image)
            
            # Hiển thị số vùng được phát hiện
            if detection_count > 0:
                self.gui.log_info(f"Phát hiện {detection_count} khuôn mặt/đầu trong ảnh")
            else:
                self.gui.log_info("Không phát hiện khuôn mặt/đầu nào trong ảnh")
            
            # Lưu ảnh kết quả
            output_path = f"head_result_{os.path.basename(file_path)}"
            cv2.imwrite(output_path, annotated_image)
            self.gui.log_info(f"Đã lưu kết quả: {output_path}")
            
        except Exception as e:
            self.gui.show_error("Lỗi", f"Lỗi khi xử lý ảnh: {str(e)}")
            self.gui.log_info(f"Lỗi: {str(e)}")
    
    def detect_from_video(self):
        """Nhận diện từ video"""
        if not self.detector.is_model_loaded():
            self.gui.show_error("Lỗi", "Model chưa được tải!")
            return
        
        # Chọn file video
        file_path = self.gui.ask_file_path(
            "Chọn video", 
            AppConfig.get_file_filters()['videos']
        )
        
        if not file_path:
            return
        
        try:
            # Mở video
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                self.gui.show_error("Lỗi", "Không thể mở file video!")
                return
            
            # Lấy thông tin video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.gui.log_info(f"Đang xử lý video: {os.path.basename(file_path)}")
            self.gui.log_info(f"Kích thước: {width}x{height}, FPS: {fps}")
            
            # Tạo video writer cho output
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
                
                # Thực hiện detection
                annotated_frame, detection_count, regions = self.detector.detect_faces_and_heads(frame.copy())
                
                # Chụp tất cả vùng phát hiện
                for region_pos in regions:
                    filepath = self.capture_manager.save_head_image(frame, region_pos)
                    if filepath:
                        captured_heads += 1
                        self.gui.log_info(f"Đã chụp từ video: {os.path.basename(filepath)}")
                
                # Ghi frame vào output video
                out.write(annotated_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Cập nhật tiến độ mỗi 30 frame
                    progress = (frame_count / total_frames) * 100
                    self.gui.log_info(f"Tiến độ: {progress:.1f}% ({frame_count}/{total_frames}) - Đã chụp {captured_heads} vùng")
            
            # Giải phóng resources
            cap.release()
            out.release()
            
            self.gui.log_info(f"Hoàn thành xử lý video! Đã lưu: {output_path}")
            self.gui.log_info(f"Tổng cộng đã chụp {captured_heads} vùng từ video")
            
        except Exception as e:
            self.gui.show_error("Lỗi", f"Lỗi khi xử lý video: {str(e)}")
            self.gui.log_info(f"Lỗi: {str(e)}")
    
    def change_save_folder(self):
        """Thay đổi thư mục lưu ảnh"""
        folder = self.gui.ask_directory("Chọn thư mục lưu ảnh")
        if folder:
            self.capture_manager.update_save_folder(folder)
            self.gui.update_folder_display(folder)
            self.gui.log_info(f"Đã thay đổi thư mục lưu: {folder}")
    
    def reset_captured_history(self):
        """Reset lịch sử khuôn mặt đã chụp"""
        self.capture_manager.reset_capture_timer()
        self.gui.log_info("Đã reset lịch sử chụp ảnh")


def main():
    """Hàm main của ứng dụng"""
    root = tk.Tk()
    app = HeadDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
