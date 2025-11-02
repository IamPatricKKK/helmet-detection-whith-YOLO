"""
Script chính để chạy toàn bộ pipeline nhận diện mũ bảo hiểm
Tự động hóa quy trình từ thu thập dữ liệu đến inference
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


class HelmetDetectionPipeline:
    """Pipeline hoàn chỉnh cho nhận diện mũ bảo hiểm"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_folder = self.project_root / "data_collection"
        self.processed_folder = self.project_root / "data_preprocessing"
        self.models_folder = self.project_root / "models"
        
    def run_data_collection(self):
        """Chạy ứng dụng thu thập dữ liệu"""
        print("=== BƯỚC 1: THU THẬP DỮ LIỆU ===")
        print("Đang mở ứng dụng thu thập dữ liệu...")
        print("Hướng dẫn:")
        print("1. Bắt đầu Camera")
        print("2. Chụp ảnh có mũ bảo hiểm")
        print("3. Chụp ảnh không có mũ")
        print("4. Thu thập ít nhất 200-500 ảnh mỗi loại")
        print("5. Đóng ứng dụng khi hoàn thành")
        
        try:
            subprocess.run([
                sys.executable, 
                str(self.project_root / "data_collection" / "data_collection_app.py")
            ])
            print("✅ Hoàn thành thu thập dữ liệu!")
        except Exception as e:
            print(f"❌ Lỗi khi chạy data collection: {e}")
            return False
        
        return True
    
    def run_preprocessing(self):
        """Chạy preprocessing dữ liệu"""
        print("\n=== BƯỚC 2: PREPROCESSING DỮ LIỆU ===")
        
        try:
            subprocess.run([
                sys.executable, 
                str(self.project_root / "data_preprocessing" / "preprocess_data.py")
            ])
            print("✅ Hoàn thành preprocessing!")
        except Exception as e:
            print(f"❌ Lỗi khi chạy preprocessing: {e}")
            return False
        
        return True
    
    def run_training(self, mode="quick", epochs=20):
        """Chạy training model"""
        print(f"\n=== BƯỚC 3: TRAINING MODEL ({mode.upper()}) ===")
        
        try:
            subprocess.run([
                sys.executable, 
                str(self.project_root / "training" / "train_model.py"),
                "--mode", mode,
                "--epochs", str(epochs)
            ])
            print("✅ Hoàn thành training!")
        except Exception as e:
            print(f"❌ Lỗi khi chạy training: {e}")
            return False
        
        return True
    
    def run_inference(self):
        """Chạy ứng dụng inference"""
        print("\n=== BƯỚC 4: INFERENCE ===")
        print("Đang mở ứng dụng inference...")
        print("Hướng dẫn:")
        print("1. Bắt đầu Camera để nhận diện real-time")
        print("2. Hoặc Test từ Ảnh để upload ảnh")
        print("3. Xem kết quả và thống kê")
        
        try:
            subprocess.run([
                sys.executable, 
                str(self.project_root / "inference" / "inference_app.py")
            ])
            print("✅ Hoàn thành inference!")
        except Exception as e:
            print(f"❌ Lỗi khi chạy inference: {e}")
            return False
        
        return True
    
    def check_dependencies(self):
        """Kiểm tra dependencies"""
        print("=== KIỂM TRA DEPENDENCIES ===")
        
        required_packages = [
            'tensorflow', 'cv2', 'numpy', 'matplotlib', 
            'sklearn', 'PIL', 'tkinter'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'PIL':
                    from PIL import Image
                elif package == 'tkinter':
                    import tkinter
                else:
                    __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n❌ Thiếu packages: {', '.join(missing_packages)}")
            print("Hãy chạy: pip install -r requirements.txt")
            return False
        
        print("✅ Tất cả dependencies đã sẵn sàng!")
        return True
    
    def check_data_exists(self):
        """Kiểm tra dữ liệu có tồn tại không"""
        helmet_folder = self.data_folder / "with_helmet"
        no_helmet_folder = self.data_folder / "no_helmet"
        
        helmet_count = len(list(helmet_folder.glob("*.jpg"))) if helmet_folder.exists() else 0
        no_helmet_count = len(list(no_helmet_folder.glob("*.jpg"))) if no_helmet_folder.exists() else 0
        
        print(f"Dữ liệu hiện tại:")
        print(f"- Có mũ bảo hiểm: {helmet_count} ảnh")
        print(f"- Không có mũ: {no_helmet_count} ảnh")
        
        if helmet_count < 50 or no_helmet_count < 50:
            print("⚠️  Khuyến nghị thu thập ít nhất 50 ảnh mỗi loại")
            return False
        
        return True
    
    def check_model_exists(self):
        """Kiểm tra model có tồn tại không"""
        model_path = self.models_folder / "helmet_detection_model.h5"
        
        if model_path.exists():
            print("✅ Model đã tồn tại")
            return True
        else:
            print("❌ Model chưa tồn tại")
            return False
    
    def run_full_pipeline(self, training_mode="quick", epochs=20):
        """Chạy toàn bộ pipeline"""
        print("🚀 BẮT ĐẦU HELMET DETECTION PIPELINE")
        print("=" * 50)
        
        # Kiểm tra dependencies
        if not self.check_dependencies():
            return False
        
        # Kiểm tra dữ liệu
        if not self.check_data_exists():
            print("\n📸 Bắt đầu thu thập dữ liệu...")
            if not self.run_data_collection():
                return False
        
        # Preprocessing
        if not self.run_preprocessing():
            return False
        
        # Training
        if not self.check_model_exists():
            if not self.run_training(training_mode, epochs):
                return False
        
        # Inference
        if not self.run_inference():
            return False
        
        print("\n🎉 HOÀN THÀNH TOÀN BỘ PIPELINE!")
        print("Model đã sẵn sàng để sử dụng!")
        
        return True
    
    def quick_start(self):
        """Quick start - chỉ chạy inference nếu model đã có"""
        print("⚡ QUICK START - INFERENCE ONLY")
        
        if not self.check_dependencies():
            return False
        
        if not self.check_model_exists():
            print("❌ Model chưa tồn tại!")
            print("Hãy chạy full pipeline trước:")
            print("python main.py --mode full")
            return False
        
        return self.run_inference()
    
    def data_only(self):
        """Chỉ thu thập dữ liệu"""
        print("📸 DATA COLLECTION ONLY")
        
        if not self.check_dependencies():
            return False
        
        return self.run_data_collection()
    
    def train_only(self, mode="quick", epochs=20):
        """Chỉ training"""
        print("🏋️ TRAINING ONLY")
        
        if not self.check_dependencies():
            return False
        
        if not self.check_data_exists():
            print("❌ Chưa có dữ liệu! Hãy thu thập trước.")
            return False
        
        # Preprocessing
        if not self.run_preprocessing():
            return False
        
        # Training
        return self.run_training(mode, epochs)


def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description='Helmet Detection Pipeline')
    parser.add_argument('--mode', choices=['full', 'quick', 'data', 'train'], 
                       default='full', help='Chế độ chạy')
    parser.add_argument('--training-mode', choices=['quick', 'full', 'transfer', 'mobile'], 
                       default='quick', help='Chế độ training')
    parser.add_argument('--epochs', type=int, default=20, help='Số epochs training')
    
    args = parser.parse_args()
    
    # Khởi tạo pipeline
    pipeline = HelmetDetectionPipeline()
    
    # Chọn mode
    if args.mode == 'full':
        success = pipeline.run_full_pipeline(args.training_mode, args.epochs)
    elif args.mode == 'quick':
        success = pipeline.quick_start()
    elif args.mode == 'data':
        success = pipeline.data_only()
    elif args.mode == 'train':
        success = pipeline.train_only(args.training_mode, args.epochs)
    
    if success:
        print("\n✅ Pipeline hoàn thành thành công!")
    else:
        print("\n❌ Pipeline gặp lỗi!")
        sys.exit(1)


if __name__ == "__main__":
    main()


