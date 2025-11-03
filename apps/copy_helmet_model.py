"""
Script để copy model nhận diện mũ bảo hiểm vào thư mục chính
"""

import os
import shutil
import sys
from pathlib import Path

# Thêm root folder vào path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
os.chdir(root_dir)  # Change working directory to root


def copy_helmet_model():
    """Copy model nhận diện mũ bảo hiểm"""
    
    # Đường dẫn source và destination
    source_model = Path("helmet_detection_project/models/helmet_detection_model.h5")
    dest_model = Path("models/helmet_detection_model.h5")
    
    # Tạo thư mục models nếu chưa có
    dest_model.parent.mkdir(exist_ok=True)
    
    if source_model.exists():
        # Copy model
        shutil.copy2(source_model, dest_model)
        print(f"✅ Đã copy model từ {source_model} đến {dest_model}")
        return True
    else:
        print(f"❌ Không tìm thấy model tại {source_model}")
        print("Hãy train model trước:")
        print("cd helmet_detection_project")
        print("python training/train_model.py --mode quick --epochs 20")
        return False


def check_dependencies():
    """Kiểm tra dependencies"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        return True
    except ImportError:
        print("❌ TensorFlow chưa được cài đặt")
        print("Hãy chạy: pip install tensorflow")
        return False


def main():
    """Hàm main"""
    print("=== COPY HELMET DETECTION MODEL ===")
    
    # Kiểm tra dependencies
    if not check_dependencies():
        return
    
    # Copy model
    if copy_helmet_model():
        print("\n🎉 Hoàn thành!")
        print("Bây giờ bạn có thể chạy:")
        print("python apps/head_detection_app_with_helmet.py")
        print("Hoặc: run_helmet_detection.bat")
    else:
        print("\n❌ Chưa thể copy model")
        print("Hãy train model trước!")


if __name__ == "__main__":
    main()


