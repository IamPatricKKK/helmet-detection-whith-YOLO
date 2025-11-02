"""
Module xử lý chụp và lưu ảnh
Tách logic capture từ main app để dễ bảo trì và mở rộng
"""

import cv2
import os
import time
from datetime import datetime


class ImageCaptureManager:
    """Class chuyên xử lý chụp và lưu ảnh"""
    
    def __init__(self, save_folder):
        self.save_folder = save_folder
        self.last_capture_time = 0
        self.stability_wait_time = 2.0  # Thời gian đợi giữa các lần chụp (giây)
        
        # Đảm bảo thư mục tồn tại
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
    
    def should_capture(self, detection_count):
        """
        Kiểm tra có nên chụp ảnh không - chụp mỗi 2 giây khi có khuôn mặt
        
        Args:
            detection_count: Số lượng phát hiện hiện tại
            
        Returns:
            tuple: (should_capture, reason)
        """
        current_time = time.time()
        
        if detection_count > 0:
            # Có phát hiện khuôn mặt, kiểm tra cooldown
            if current_time - self.last_capture_time >= self.stability_wait_time:
                return True, "Chụp định kỳ mỗi 2 giây"
        
        return False, ""
    
    def save_head_image(self, frame, head_position):
        """
        Lưu ảnh đầu người với cải tiến
        
        Args:
            frame: Frame ảnh gốc
            head_position: Vị trí đầu người (x, y, w, h)
            
        Returns:
            str: Đường dẫn file đã lưu, None nếu lỗi
        """
        x, y, w, h = head_position
        
        # Mở rộng bounding box để bao gồm cả đầu và cổ
        margin_x = int(w * 0.2)  # Tăng margin ngang
        margin_y = int(h * 0.25)  # Tăng margin dọc để bao gồm cổ
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + w + margin_x)
        y2 = min(frame.shape[0], y + h + margin_y)
        
        # Đảm bảo kích thước tối thiểu và tỷ lệ hợp lý
        if (x2 - x1) < 150 or (y2 - y1) < 150:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            size = max(150, max(x2 - x1, y2 - y1))
            x1 = max(0, center_x - size // 2)
            y1 = max(0, center_y - size // 2)
            x2 = min(frame.shape[1], center_x + size // 2)
            y2 = min(frame.shape[0], center_y + size // 2)
        
        # Cắt ảnh đầu người
        head_img = frame[y1:y2, x1:x2]
        
        # Kiểm tra xem ảnh có hợp lệ không
        if head_img.size == 0:
            return None
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"head_{timestamp}.jpg"
        filepath = os.path.join(self.save_folder, filename)
        
        # Lưu ảnh với chất lượng cao
        success = cv2.imwrite(filepath, head_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            # Cập nhật thời gian chụp cuối
            self.last_capture_time = time.time()
            return filepath
        else:
            return None
    
    def get_remaining_time(self):
        """
        Lấy thời gian còn lại đến lần chụp tiếp theo
        
        Returns:
            float: Thời gian còn lại (giây)
        """
        if self.last_capture_time == 0:
            return 0
        
        current_time = time.time()
        remaining = max(0, self.stability_wait_time - (current_time - self.last_capture_time))
        return remaining
    
    def update_save_folder(self, new_folder):
        """
        Cập nhật thư mục lưu ảnh
        
        Args:
            new_folder: Đường dẫn thư mục mới
        """
        self.save_folder = new_folder
        
        # Đảm bảo thư mục tồn tại
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
    
    def reset_capture_timer(self):
        """Reset timer chụp ảnh"""
        self.last_capture_time = 0
    
    def set_capture_interval(self, seconds):
        """
        Thiết lập khoảng thời gian giữa các lần chụp
        
        Args:
            seconds: Số giây giữa các lần chụp
        """
        self.stability_wait_time = seconds
