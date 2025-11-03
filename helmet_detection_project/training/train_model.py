"""
Script training model nhận diện mũ bảo hiểm
Kết hợp tất cả các bước từ load data đến train model
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Thêm đường dẫn để import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.helmet_model import HelmetDetectionModel
from data_preprocessing.preprocess_data import DataPreprocessor


class HelmetTrainingPipeline:
    """Pipeline hoàn chỉnh để train model nhận diện mũ bảo hiểm"""
    
    def __init__(self, data_folder="data_collection", processed_folder="data_preprocessing"):
        self.data_folder = data_folder
        self.processed_folder = processed_folder
        self.model = None
        self.metadata = None
        
    def run_full_pipeline(self, model_type="custom", epochs=50, batch_size=32, 
                         use_augmentation=True, augmentation_factor=2):
        """
        Chạy pipeline hoàn chỉnh từ preprocessing đến training
        
        Args:
            model_type: Loại model ("custom", "resnet", "mobilenet")
            epochs: Số epochs training
            batch_size: Batch size
            use_augmentation: Có sử dụng data augmentation không
            augmentation_factor: Factor cho data augmentation
        """
        print("=== HELMET DETECTION TRAINING PIPELINE ===")
        
        # Bước 1: Kiểm tra dữ liệu đã được preprocess chưa
        if not self._check_processed_data():
            print("Dữ liệu chưa được preprocess. Đang chạy preprocessing...")
            self._run_preprocessing(use_augmentation, augmentation_factor)
        
        # Bước 2: Load dữ liệu đã được preprocess
        print("\nĐang load dữ liệu đã được preprocess...")
        X_train, X_val, X_test, y_train, y_val, y_test = self._load_processed_data()
        
        # Bước 3: Tạo và compile model
        print("\nĐang tạo model...")
        self.model = HelmetDetectionModel()
        # Sử dụng improved features mặc định (multi-branch cho custom, attention cho transfer learning)
        use_improved = True
        if model_type == "multi_branch":
            self.model.create_model(model_type="multi_branch", use_improved_features=True)
        else:
            self.model.create_model(model_type=model_type, use_improved_features=use_improved)
        self.model.compile_model(learning_rate=0.001)
        
        # Hiển thị model summary
        print("\nModel Summary:")
        self.model.model.summary()
        
        # Bước 4: Train model
        print(f"\nBắt đầu training với {epochs} epochs...")
        self.model.train_model(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size
        )
        
        # Bước 5: Đánh giá model
        print("\nĐang đánh giá model...")
        evaluation_results = self.model.evaluate_model(X_test, y_test)
        
        # Bước 6: Vẽ đồ thị training history
        print("\nĐang vẽ đồ thị training history...")
        self.model.plot_training_history()
        
        # Bước 7: Lưu model
        print("\nĐang lưu model...")
        self.model.save_model("models/helmet_detection_model.h5")
        
        # Bước 8: Lưu kết quả training
        self._save_training_results(evaluation_results)
        
        print("\n✅ Hoàn thành training pipeline!")
        print(f"Model đã được lưu tại: models/helmet_detection_model.h5")
        print(f"Test accuracy: {evaluation_results['test_accuracy']:.4f}")
        
        return evaluation_results
    
    def _check_processed_data(self):
        """Kiểm tra xem dữ liệu đã được preprocess chưa"""
        required_files = [
            "X_train.npy", "X_val.npy", "X_test.npy",
            "y_train.npy", "y_val.npy", "y_test.npy",
            "metadata.pkl"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(self.processed_folder, file)):
                return False
        return True
    
    def _run_preprocessing(self, use_augmentation=True, augmentation_factor=2):
        """Chạy preprocessing"""
        preprocessor = DataPreprocessor(self.data_folder)
        
        # Load và preprocess dữ liệu
        images, labels = preprocessor.load_and_preprocess_images()
        
        if len(images) == 0:
            raise ValueError("Không tìm thấy dữ liệu! Hãy chạy data_collection_app.py trước.")
        
        # Visualize phân bố dữ liệu
        preprocessor.visualize_data_distribution(labels, self.processed_folder)
        
        # Chia dữ liệu
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(images, labels)
        
        # Tăng cường dữ liệu nếu được yêu cầu
        if use_augmentation:
            print(f"Đang tăng cường dữ liệu với factor {augmentation_factor}...")
            X_train, y_train = preprocessor.augment_data(X_train, y_train, augmentation_factor)
        
        # Lưu dữ liệu đã xử lý
        preprocessor.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, self.processed_folder)
    
    def _load_processed_data(self):
        """Load dữ liệu đã được preprocess"""
        # Load arrays
        X_train = np.load(os.path.join(self.processed_folder, "X_train.npy"))
        X_val = np.load(os.path.join(self.processed_folder, "X_val.npy"))
        X_test = np.load(os.path.join(self.processed_folder, "X_test.npy"))
        y_train = np.load(os.path.join(self.processed_folder, "y_train.npy"))
        y_val = np.load(os.path.join(self.processed_folder, "y_val.npy"))
        y_test = np.load(os.path.join(self.processed_folder, "y_test.npy"))
        
        # Load metadata
        with open(os.path.join(self.processed_folder, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded data:")
        print(f"- Train: {len(X_train)} samples")
        print(f"- Validation: {len(X_val)} samples")
        print(f"- Test: {len(X_test)} samples")
        print(f"- Image shape: {X_train[0].shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _save_training_results(self, evaluation_results):
        """Lưu kết quả training"""
        results = {
            'test_accuracy': evaluation_results['test_accuracy'],
            'test_loss': evaluation_results['test_loss'],
            'classification_report': evaluation_results['classification_report'],
            'confusion_matrix': evaluation_results['confusion_matrix'].tolist(),
            'metadata': self.metadata
        }
        
        with open("models/training_results.pkl", "wb") as f:
            pickle.dump(results, f)
        
        print("Đã lưu kết quả training vào models/training_results.pkl")
    
    def quick_train(self, epochs=20):
        """
        Training nhanh với ít epochs để test
        
        Args:
            epochs: Số epochs (mặc định 20)
        """
        print("=== QUICK TRAINING MODE ===")
        print("Chế độ này sử dụng để test nhanh với ít epochs")
        
        return self.run_full_pipeline(
            model_type="custom",
            epochs=epochs,
            batch_size=16,
            use_augmentation=False
        )
    
    def train_with_transfer_learning(self, epochs=30):
        """
        Training với transfer learning (ResNet50)
        
        Args:
            epochs: Số epochs
        """
        print("=== TRANSFER LEARNING MODE ===")
        print("Sử dụng ResNet50 pre-trained model")
        
        return self.run_full_pipeline(
            model_type="resnet",
            epochs=epochs,
            batch_size=16,
            use_augmentation=True,
            augmentation_factor=3
        )
    
    def train_mobile_model(self, epochs=40):
        """
        Training với MobileNetV2 (nhẹ hơn, phù hợp mobile)
        
        Args:
            epochs: Số epochs
        """
        print("=== MOBILE MODEL MODE ===")
        print("Sử dụng MobileNetV2 cho ứng dụng mobile")
        
        return self.run_full_pipeline(
            model_type="mobilenet",
            epochs=epochs,
            batch_size=32,
            use_augmentation=True,
            augmentation_factor=2
        )


def main():
    """Hàm main để chạy training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train helmet detection model')
    parser.add_argument('--mode', choices=['quick', 'full', 'transfer', 'mobile'], 
                       default='quick', help='Training mode')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Khởi tạo pipeline
    pipeline = HelmetTrainingPipeline()
    
    # Chọn mode training
    if args.mode == 'quick':
        results = pipeline.quick_train(epochs=args.epochs)
    elif args.mode == 'full':
        results = pipeline.run_full_pipeline(epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == 'transfer':
        results = pipeline.train_with_transfer_learning(epochs=args.epochs)
    elif args.mode == 'mobile':
        results = pipeline.train_mobile_model(epochs=args.epochs)
    
    print(f"\n🎉 Training hoàn thành!")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Model đã sẵn sàng để sử dụng!")


if __name__ == "__main__":
    main()
