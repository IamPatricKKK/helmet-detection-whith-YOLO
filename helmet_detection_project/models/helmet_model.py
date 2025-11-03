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
        
    def create_model(self, model_type="custom", use_improved_features=True):
        """
        Tạo model CNN
        
        Args:
            model_type: Loại model ("custom", "resnet", "mobilenet", "multi_branch")
            use_improved_features: Sử dụng multi-branch với HSV và spatial attention
        """
        if model_type == "custom":
            if use_improved_features:
                self.model = self._create_improved_custom_model()
            else:
                self.model = self._create_custom_model()
        elif model_type == "resnet":
            if use_improved_features:
                self.model = self._create_improved_resnet_model()
            else:
                self.model = self._create_resnet_model()
        elif model_type == "mobilenet":
            if use_improved_features:
                self.model = self._create_improved_mobilenet_model()
            else:
                self.model = self._create_mobilenet_model()
        elif model_type == "multi_branch":
            self.model = self._create_multi_branch_model()
        else:
            raise ValueError("model_type phải là 'custom', 'resnet', 'mobilenet', hoặc 'multi_branch'")
        
        print(f"Đã tạo model {model_type} (improved_features={use_improved_features})")
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
    
    def _create_multi_branch_model(self):
        """
        Tạo multi-branch model với RGB và HSV branches ⭐ CẢI TIẾN
        Sử dụng cả RGB và HSV color spaces để cải thiện accuracy
        """
        # Input RGB
        input_rgb = layers.Input(shape=self.input_shape, name='rgb_input')
        
        # Input HSV
        input_hsv = layers.Input(shape=self.input_shape, name='hsv_input')
        
        # Branch 1: RGB CNN
        rgb_branch = self._create_cnn_branch(input_rgb, name='rgb')
        
        # Branch 2: HSV CNN
        hsv_branch = self._create_cnn_branch(input_hsv, name='hsv')
        
        # Concatenate branches
        concatenated = layers.Concatenate(name='concatenate_branches')([rgb_branch, hsv_branch])
        
        # Spatial Attention (tập trung vào vùng quan trọng)
        attention = self._create_spatial_attention(concatenated)
        
        # Dense layers
        x = layers.Dense(512, activation='relu', name='dense1')(attention)
        x = layers.BatchNormalization(name='bn_dense1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        
        x = layers.Dense(256, activation='relu', name='dense2')(x)
        x = layers.BatchNormalization(name='bn_dense2')(x)
        x = layers.Dropout(0.5, name='dropout2')(x)
        
        # Output
        output = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs=[input_rgb, input_hsv], outputs=output, name='multi_branch_helmet_model')
        
        return model
    
    def _create_cnn_branch(self, input_layer, name=''):
        """Tạo một CNN branch cho multi-branch model"""
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name=f'{name}_conv1')(input_layer)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.MaxPooling2D((2, 2), name=f'{name}_pool1')(x)
        x = layers.Dropout(0.25, name=f'{name}_drop1')(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        x = layers.MaxPooling2D((2, 2), name=f'{name}_pool2')(x)
        x = layers.Dropout(0.25, name=f'{name}_drop2')(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name=f'{name}_conv3')(x)
        x = layers.BatchNormalization(name=f'{name}_bn3')(x)
        x = layers.MaxPooling2D((2, 2), name=f'{name}_pool3')(x)
        x = layers.Dropout(0.25, name=f'{name}_drop3')(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name=f'{name}_conv4')(x)
        x = layers.BatchNormalization(name=f'{name}_bn4')(x)
        x = layers.MaxPooling2D((2, 2), name=f'{name}_pool4')(x)
        x = layers.Dropout(0.25, name=f'{name}_drop4')(x)
        
        # Global Average Pooling để convert từ 4D tensor sang 2D
        x = layers.GlobalAveragePooling2D(name=f'{name}_gap')(x)
        
        return x
    
    def _create_spatial_attention(self, input_tensor):
        """
        Tạo spatial attention mechanism
        Tập trung vào features quan trọng (vùng mũ bảo hiểm)
        """
        # Tính attention weights
        attention = layers.Dense(512, activation='tanh', name='attention_dense1')(input_tensor)
        attention = layers.Dense(256, activation='sigmoid', name='attention_dense2')(attention)
        
        # Apply attention
        attended = layers.Multiply(name='attention_apply')([input_tensor, attention])
        
        return attended
    
    def _create_improved_custom_model(self):
        """Tạo custom model với spatial attention"""
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Spatial Attention
        x = self._create_spatial_attention(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _create_improved_resnet_model(self):
        """Tạo ResNet model với attention"""
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        base_model.trainable = False
        
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Spatial Attention
        x = self._create_spatial_attention(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _create_improved_mobilenet_model(self):
        """Tạo MobileNet model với attention"""
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        base_model.trainable = False
        
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Spatial Attention
        x = self._create_spatial_attention(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
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
    
    def predict_single_image(self, image, use_roi_crop=True):
        """
        Dự đoán cho một ảnh đơn lẻ
        
        Args:
            image: Ảnh input (numpy array RGB hoặc BGR)
            use_roi_crop: Có crop vùng top 30% (vùng mũ bảo hiểm) không
            
        Returns:
            tuple: (prediction, confidence)
        """
        if self.model is None:
            raise ValueError("Model chưa được load!")
        
        # Kiểm tra model type
        is_multi_branch = len(self.model.inputs) == 2
        
        # Preprocess image
        if use_roi_crop and image.shape[0] > 100:
            # Crop top 30% để focus vào vùng mũ bảo hiểm
            h = image.shape[0]
            top_region = image[:int(h*0.3), :, :]
            # Resize lại về kích thước chuẩn
            import cv2
            image = cv2.resize(top_region, (self.input_shape[0], self.input_shape[1]))
        
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        if is_multi_branch:
            # Multi-branch model cần cả RGB và HSV
            rgb_image = image
            # Convert RGB to HSV
            import cv2
            if len(image.shape) == 4:
                hsv_image = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in image])
            else:
                # Convert single image
                hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                # Normalize HSV: H [0, 360] -> [0, 1], S [0, 255] -> [0, 1], V [0, 255] -> [0, 1]
                hsv_image[:, :, 0] = hsv_image[:, :, 0] / 180.0
                hsv_image[:, :, 1] = hsv_image[:, :, 1] / 255.0
                hsv_image[:, :, 2] = hsv_image[:, :, 2] / 255.0
            
            # Reshape for prediction
            if len(rgb_image.shape) == 3:
                rgb_image = np.expand_dims(rgb_image, axis=0)
            if len(hsv_image.shape) == 3:
                hsv_image = np.expand_dims(hsv_image, axis=0)
            
            # Predict
            prediction = self.model.predict([rgb_image, hsv_image], verbose=0)
        else:
            # Single input model
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
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


