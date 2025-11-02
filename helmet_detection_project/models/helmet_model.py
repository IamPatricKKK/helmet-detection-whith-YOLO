"""
Model CNN để nhận diện mũ bảo hiểm
Sử dụng TensorFlow/Keras để xây dựng model phân loại
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class HelmetDetectionModel:
    """Class chứa model CNN để nhận diện mũ bảo hiểm"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def create_model(self, model_type="custom"):
        """
        Tạo model CNN
        
        Args:
            model_type: Loại model ("custom", "resnet", "mobilenet")
        """
        if model_type == "custom":
            self.model = self._create_custom_model()
        elif model_type == "resnet":
            self.model = self._create_resnet_model()
        elif model_type == "mobilenet":
            self.model = self._create_mobilenet_model()
        else:
            raise ValueError("model_type phải là 'custom', 'resnet', hoặc 'mobilenet'")
        
        print(f"Đã tạo model {model_type}")
        print(f"Tổng số parameters: {self.model.count_params():,}")
    
    def _create_custom_model(self):
        """Tạo custom CNN model"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_resnet_model(self):
        """Tạo model dựa trên ResNet50"""
        # Load pre-trained ResNet50
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_mobilenet_model(self):
        """Tạo model dựa trên MobileNetV2"""
        # Load pre-trained MobileNetV2
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model với optimizer và loss function
        
        Args:
            learning_rate: Learning rate cho optimizer
        """
        if self.model is None:
            raise ValueError("Model chưa được tạo! Hãy gọi create_model() trước.")
        
        # Optimizer với learning rate scheduling
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Đã compile model")
        print(f"Optimizer: Adam (lr={learning_rate})")
        print(f"Loss: sparse_categorical_crossentropy")
        print(f"Metrics: accuracy")
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs=50, batch_size=32, callbacks=None):
        """
        Train model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Số epochs
            batch_size: Batch size
            callbacks: List callbacks
        """
        if self.model is None:
            raise ValueError("Model chưa được tạo và compile!")
        
        print(f"Bắt đầu training với {epochs} epochs...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        
        # Default callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Hoàn thành training!")
    
    def _get_default_callbacks(self):
        """Tạo default callbacks"""
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                'models/best_helmet_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def evaluate_model(self, X_test, y_test):
        """
        Đánh giá model trên test set
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model chưa được tạo!")
        
        print("Đang đánh giá model trên test set...")
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        class_names = ["Không có mũ", "Có mũ bảo hiểm"]
        report = classification_report(y_test, y_pred_classes, target_names=class_names)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        self._plot_confusion_matrix(cm, class_names)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Vẽ confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """Vẽ đồ thị training history"""
        if self.history is None:
            print("Chưa có training history!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath="models/helmet_detection_model.h5"):
        """
        Lưu model
        
        Args:
            filepath: Đường dẫn lưu model
        """
        if self.model is None:
            raise ValueError("Model chưa được tạo!")
        
        # Tạo thư mục nếu chưa có
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Lưu model
        self.model.save(filepath)
        print(f"✅ Đã lưu model tại: {filepath}")
    
    def load_model(self, filepath="models/helmet_detection_model.h5"):
        """
        Load model đã lưu
        
        Args:
            filepath: Đường dẫn model
        """
        self.model = keras.models.load_model(filepath)
        print(f"✅ Đã load model từ: {filepath}")
    
    def predict_single_image(self, image):
        """
        Dự đoán cho một ảnh đơn lẻ
        
        Args:
            image: Ảnh input (numpy array)
            
        Returns:
            tuple: (prediction, confidence)
        """
        if self.model is None:
            raise ValueError("Model chưa được load!")
        
        # Preprocess image
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Reshape for prediction
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Predict
        prediction = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        class_names = ["Không có mũ", "Có mũ bảo hiểm"]
        
        return class_names[predicted_class], confidence


def main():
    """Hàm main để test model"""
    print("=== HELMET DETECTION MODEL ===")
    
    # Tạo model
    model = HelmetDetectionModel()
    
    # Tạo custom model
    model.create_model("custom")
    
    # Compile model
    model.compile_model(learning_rate=0.001)
    
    # Hiển thị model summary
    model.model.summary()
    
    print("✅ Model đã sẵn sàng để training!")


if __name__ == "__main__":
    main()


