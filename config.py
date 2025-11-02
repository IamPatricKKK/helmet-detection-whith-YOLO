"""
Module quản lý cấu hình ứng dụng
Tách các thông số cấu hình để dễ thay đổi và bảo trì
"""

import os
from datetime import datetime


class AppConfig:
    """Class quản lý cấu hình ứng dụng"""
    
    # Cấu hình giao diện
    WINDOW_TITLE = "Ứng dụng Nhận diện Khuôn mặt & Đầu người"
    WINDOW_SIZE = "800x600"
    
    # Cấu hình camera
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    CAMERA_BUFFER_SIZE = 1
    
    # Cấu hình nhận diện
    HAAR_SCALE_FACTOR = 1.1
    HAAR_MIN_NEIGHBORS = 12
    HAAR_MIN_SIZE = (80, 80)
    HAAR_MAX_SIZE = (250, 250)
    
    YOLO_CONFIDENCE = 0.5
    YOLO_MIN_HEIGHT = 100
    YOLO_MIN_WIDTH = 50
    
    # Cấu hình chụp ảnh
    CAPTURE_INTERVAL = 2.0  # giây
    IMAGE_MARGIN_X = 0.2  # 20% margin ngang
    IMAGE_MARGIN_Y = 0.25  # 25% margin dọc
    MIN_IMAGE_SIZE = 150
    JPEG_QUALITY = 95
    
    # Cấu hình hiển thị
    MAX_DISPLAY_WIDTH = 600
    MAX_RESULT_WIDTH = 800
    
    # Cấu hình thư mục
    MAIN_FOLDER = "captured_heads"
    
    # Cấu hình logging
    NO_DETECTION_LOG_INTERVAL = 3.0  # giây
    
    @classmethod
    def get_session_folder(cls):
        """
        Tạo đường dẫn thư mục session mới
        
        Returns:
            str: Đường dẫn thư mục session
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = f"session_{timestamp}"
        return os.path.join(cls.MAIN_FOLDER, session_folder)
    
    @classmethod
    def ensure_main_folder(cls):
        """
        Đảm bảo thư mục chính tồn tại
        
        Returns:
            str: Đường dẫn thư mục chính
        """
        if not os.path.exists(cls.MAIN_FOLDER):
            os.makedirs(cls.MAIN_FOLDER)
        return cls.MAIN_FOLDER
    
    @classmethod
    def get_model_paths(cls):
        """
        Lấy đường dẫn các model
        
        Returns:
            dict: Dictionary chứa đường dẫn các model
        """
        return {
            'yolo': 'yolov8n.pt',
            'haar_cascade': 'haarcascade_frontalface_default.xml'
        }
    
    @classmethod
    def get_file_filters(cls):
        """
        Lấy bộ lọc file cho dialog
        
        Returns:
            dict: Dictionary chứa các bộ lọc file
        """
        return {
            'images': [("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")],
            'videos': [("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        }
    
    @classmethod
    def get_detection_params(cls):
        """
        Lấy tham số nhận diện
        
        Returns:
            dict: Dictionary chứa các tham số nhận diện
        """
        return {
            'haar': {
                'scaleFactor': cls.HAAR_SCALE_FACTOR,
                'minNeighbors': cls.HAAR_MIN_NEIGHBORS,
                'minSize': cls.HAAR_MIN_SIZE,
                'maxSize': cls.HAAR_MAX_SIZE
            },
            'yolo': {
                'conf': cls.YOLO_CONFIDENCE,
                'minHeight': cls.YOLO_MIN_HEIGHT,
                'minWidth': cls.YOLO_MIN_WIDTH
            }
        }
    
    @classmethod
    def get_capture_params(cls):
        """
        Lấy tham số chụp ảnh
        
        Returns:
            dict: Dictionary chứa các tham số chụp ảnh
        """
        return {
            'interval': cls.CAPTURE_INTERVAL,
            'marginX': cls.IMAGE_MARGIN_X,
            'marginY': cls.IMAGE_MARGIN_Y,
            'minSize': cls.MIN_IMAGE_SIZE,
            'quality': cls.JPEG_QUALITY
        }
