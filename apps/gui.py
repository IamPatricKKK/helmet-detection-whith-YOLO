"""
Module xử lý giao diện người dùng
Tách logic GUI từ main app để dễ bảo trì và mở rộng
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from config import AppConfig


class AppGUI:
    """Class chuyên xử lý giao diện người dùng"""
    
    def __init__(self, root, app_instance):
        self.root = root
        self.app = app_instance
        self._setup_window()
        self._create_widgets()
    
    def _setup_window(self):
        """Thiết lập cửa sổ chính"""
        self.root.title(AppConfig.WINDOW_TITLE)
        self.root.geometry(AppConfig.WINDOW_SIZE)
        
        # Cấu hình grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def _create_widgets(self):
        """Tạo các widget giao diện"""
        # Frame chính
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tiêu đề
        self._create_title()
        
        # Frame chọn chế độ
        self._create_mode_frame()
        
        # Frame cài đặt
        self._create_settings_frame()
        
        # Frame hiển thị kết quả
        self._create_result_frame()
        
        # Frame thông tin
        self._create_info_frame()
        
        # Cấu hình grid weights
        self._configure_grid_weights()
    
    def _create_title(self):
        """Tạo tiêu đề ứng dụng"""
        title_label = ttk.Label(
            self.main_frame, 
            text=AppConfig.WINDOW_TITLE, 
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
    
    def _create_mode_frame(self):
        """Tạo frame chọn chế độ"""
        mode_frame = ttk.LabelFrame(
            self.main_frame, 
            text="Chọn chế độ nhận diện", 
            padding="10"
        )
        mode_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Buttons
        self.camera_btn = ttk.Button(
            mode_frame, 
            text="Bắt đầu Camera", 
            command=self.app.start_camera_detection
        )
        self.camera_btn.grid(row=0, column=0, padx=5)
        
        self.image_btn = ttk.Button(
            mode_frame, 
            text="Nhận diện từ Ảnh", 
            command=self.app.detect_from_image
        )
        self.image_btn.grid(row=0, column=1, padx=5)
        
        self.video_btn = ttk.Button(
            mode_frame, 
            text="Nhận diện từ Video", 
            command=self.app.detect_from_video
        )
        self.video_btn.grid(row=0, column=2, padx=5)
        
        self.stop_btn = ttk.Button(
            mode_frame, 
            text="Dừng Camera", 
            command=self.app.stop_camera, 
            state="disabled"
        )
        self.stop_btn.grid(row=0, column=3, padx=5)
    
    def _create_settings_frame(self):
        """Tạo frame cài đặt"""
        settings_frame = ttk.LabelFrame(
            self.main_frame, 
            text="Cài đặt", 
            padding="10"
        )
        settings_frame.grid(row=1, column=2, sticky=(tk.W, tk.E), padx=(10, 0))
        
        # Thư mục lưu
        ttk.Label(settings_frame, text="Thư mục lưu:").grid(row=0, column=0, padx=5)
        self.folder_label = ttk.Label(
            settings_frame, 
            text=self.app.capture_manager.save_folder, 
            foreground="blue"
        )
        self.folder_label.grid(row=0, column=1, padx=5)
        
        ttk.Button(
            settings_frame, 
            text="Đổi thư mục", 
            command=self.app.change_save_folder
        ).grid(row=0, column=2, padx=5)
        
        # Reset lịch sử
        ttk.Button(
            settings_frame, 
            text="Reset lịch sử", 
            command=self.app.reset_captured_history
        ).grid(row=1, column=0, columnspan=3, pady=5)
    
    def _create_result_frame(self):
        """Tạo frame hiển thị kết quả"""
        result_frame = ttk.LabelFrame(
            self.main_frame, 
            text="Kết quả nhận diện khuôn mặt & đầu người", 
            padding="10"
        )
        result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Label hiển thị ảnh
        self.image_label = ttk.Label(
            result_frame, 
            text="Chưa có dữ liệu", 
            background="white", 
            anchor="center"
        )
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def _create_info_frame(self):
        """Tạo frame thông tin"""
        info_frame = ttk.LabelFrame(
            self.main_frame, 
            text="Thông tin", 
            padding="10"
        )
        info_frame.grid(row=2, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                       padx=(10, 0), pady=(10, 0))
        
        # Text widget cho thông tin
        self.info_text = tk.Text(info_frame, height=15, width=30, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def _configure_grid_weights(self):
        """Cấu hình grid weights"""
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(0, weight=1)
        self.info_frame.columnconfigure(0, weight=1)
        self.info_frame.rowconfigure(0, weight=1)
    
    def log_info(self, message):
        """Ghi thông tin vào text widget"""
        self.info_text.insert(tk.END, f"{message}\n")
        self.info_text.see(tk.END)
        self.root.update()
    
    def update_image_display(self, frame):
        """
        Cập nhật hiển thị ảnh
        
        Args:
            frame: Frame ảnh cần hiển thị
        """
        # Chuyển đổi từ BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame để hiển thị
        height, width = rgb_frame.shape[:2]
        max_width = AppConfig.MAX_DISPLAY_WIDTH
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
        
        # Chuyển đổi sang PIL Image
        pil_image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Cập nhật giao diện
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def update_folder_display(self, folder_path):
        """
        Cập nhật hiển thị thư mục
        
        Args:
            folder_path: Đường dẫn thư mục mới
        """
        self.folder_label.config(text=folder_path)
    
    def set_camera_buttons_state(self, camera_running):
        """
        Thiết lập trạng thái các button camera
        
        Args:
            camera_running: True nếu camera đang chạy
        """
        if camera_running:
            self.camera_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
        else:
            self.camera_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
    
    def show_error(self, title, message):
        """
        Hiển thị dialog lỗi
        
        Args:
            title: Tiêu đề dialog
            message: Nội dung thông báo
        """
        messagebox.showerror(title, message)
    
    def ask_file_path(self, title, file_types):
        """
        Hiển thị dialog chọn file
        
        Args:
            title: Tiêu đề dialog
            file_types: Danh sách loại file
            
        Returns:
            str: Đường dẫn file được chọn
        """
        return filedialog.askopenfilename(title=title, filetypes=file_types)
    
    def ask_directory(self, title):
        """
        Hiển thị dialog chọn thư mục
        
        Args:
            title: Tiêu đề dialog
            
        Returns:
            str: Đường dẫn thư mục được chọn
        """
        return filedialog.askdirectory(title=title)
