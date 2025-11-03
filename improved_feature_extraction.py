"""
Improved Feature Extraction cho Nhận diện Mũ Bảo Hiểm
Tích hợp nhiều loại features: Color, Texture, Spatial, Multi-scale
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from skimage.feature import local_binary_pattern, hog
from skimage import exposure


class ImprovedHelmetFeatureExtractor:
    """Class trích xuất nhiều loại features cho nhận diện mũ bảo hiểm"""
    
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
    
    def extract_color_features(self, image):
        """
        Trích xuất đặc trưng màu sắc từ nhiều không gian màu
        
        Args:
            image: RGB image (224, 224, 3)
            
        Returns:
            dict: Dictionary chứa các color features
        """
        features = {}
        
        # 1. RGB Histogram (đã có)
        for i, color in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            features[f'rgb_{color}_hist'] = hist.flatten()[:64]  # Reduce dimension
        
        # 2. HSV Features ⭐ QUAN TRỌNG
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Histogram của Hue (màu sắc) - mũ bảo hiểm thường có màu đặc trưng
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        features['hue_hist'] = h_hist.flatten()[:32]
        features['saturation_hist'] = s_hist.flatten()[:32]
        features['value_hist'] = v_hist.flatten()[:32]
        
        # 3. LAB Color Space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_hist = cv2.calcHist([lab], [0], None, [256], [0, 256])
        features['lab_l_hist'] = l_hist.flatten()[:32]
        
        # 4. Mean và Std của từng channel (mô tả màu sắc tổng thể)
        for i, color_space in enumerate(['rgb', 'hsv', 'lab']):
            if color_space == 'rgb':
                img = image
            elif color_space == 'hsv':
                img = hsv
            else:
                img = lab
            
            for j, channel in enumerate(['ch0', 'ch1', 'ch2']):
                features[f'{color_space}_{channel}_mean'] = np.mean(img[:, :, j])
                features[f'{color_space}_{channel}_std'] = np.std(img[:, :, j])
        
        return features
    
    def extract_texture_features(self, image):
        """
        Trích xuất đặc trưng texture
        
        Args:
            image: RGB image (224, 224, 3)
            
        Returns:
            dict: Dictionary chứa texture features
        """
        features = {}
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Local Binary Pattern (LBP) - Mô tả texture địa phương
        # Mũ bảo hiểm có texture khác với tóc/da
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Histogram của LBP
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # Normalize
        features['lbp_hist'] = hist
        
        # 2. HOG (Histogram of Oriented Gradients) - Mô tả shape và texture
        # Chỉ lấy một số features quan trọng
        hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), feature_vector=True)
        # Giảm dimension
        features['hog'] = hog_features[::10]  # Sample every 10th feature
        
        # 3. GLCM (Gray-Level Co-occurrence Matrix) - Texture statistics
        # Tính contrast, correlation, energy, homogeneity
        from skimage.feature import graycomatrix, graycoprops
        
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=256, symmetric=True, normed=True)
        
        for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
            features[f'glcm_{prop}'] = graycoprops(glcm, prop)[0, 0]
        
        return features
    
    def extract_spatial_features(self, image, face_bbox=None):
        """
        Trích xuất đặc trưng không gian - tập trung vào vùng đầu
        
        Args:
            image: RGB image (224, 224, 3)
            face_bbox: (x, y, w, h) của face bounding box
            
        Returns:
            dict: Dictionary chứa spatial features
        """
        features = {}
        h, w = image.shape[:2]
        
        # 1. Vùng ROI: Top 30% của ảnh (nơi đeo mũ bảo hiểm)
        top_region = image[:int(h*0.3), :, :]
        middle_region = image[int(h*0.3):int(h*0.7), :, :]
        bottom_region = image[int(h*0.7):, :, :]
        
        # 2. Color features của từng vùng
        for region_name, region in [('top', top_region), ('middle', middle_region), ('bottom', bottom_region)]:
            if region.size > 0:
                # Mean color của từng vùng
                mean_color = np.mean(region, axis=(0, 1))
                features[f'{region_name}_region_mean_r'] = mean_color[0]
                features[f'{region_name}_region_mean_g'] = mean_color[1]
                features[f'{region_name}_region_mean_b'] = mean_color[2]
                
                # Variance (mô tả sự thay đổi màu)
                var_color = np.var(region, axis=(0, 1))
                features[f'{region_name}_region_var_r'] = var_color[0]
                features[f'{region_name}_region_var_g'] = var_color[1]
                features[f'{region_name}_region_var_b'] = var_color[2]
        
        # 3. Tỷ lệ vùng trên/middle (mũ bảo hiểm làm thay đổi tỷ lệ này)
        if top_region.size > 0 and middle_region.size > 0:
            top_brightness = np.mean(cv2.cvtColor(top_region, cv2.COLOR_RGB2GRAY))
            middle_brightness = np.mean(cv2.cvtColor(middle_region, cv2.COLOR_RGB2GRAY))
            features['top_middle_brightness_ratio'] = top_brightness / (middle_brightness + 1e-7)
        
        # 4. Edge density trong vùng top (mũ bảo hiểm có nhiều edges)
        if top_region.size > 0:
            gray_top = cv2.cvtColor(top_region, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_top, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.size + 1e-7)
            features['top_region_edge_density'] = edge_density
        
        return features
    
    def extract_all_features(self, image, face_bbox=None):
        """
        Trích xuất tất cả features
        
        Args:
            image: RGB image (224, 224, 3)
            face_bbox: Optional face bounding box
            
        Returns:
            np.array: Concatenated feature vector
        """
        # Extract từng loại feature
        color_features = self.extract_color_features(image)
        texture_features = self.extract_texture_features(image)
        spatial_features = self.extract_spatial_features(image, face_bbox)
        
        # Combine tất cả
        all_features = []
        
        # Color features
        for key in sorted(color_features.keys()):
            feat = color_features[key]
            if isinstance(feat, np.ndarray):
                all_features.extend(feat.tolist())
            else:
                all_features.append(feat)
        
        # Texture features
        for key in sorted(texture_features.keys()):
            feat = texture_features[key]
            if isinstance(feat, np.ndarray):
                all_features.extend(feat.tolist())
            else:
                all_features.append(feat)
        
        # Spatial features
        for key in sorted(spatial_features.keys()):
            feat = spatial_features[key]
            if isinstance(feat, np.ndarray):
                all_features.extend(feat.tolist())
            else:
                all_features.append(feat)
        
        return np.array(all_features, dtype=np.float32)
    
    def create_multi_branch_cnn(self):
        """
        Tạo CNN model với multiple branches cho các loại features khác nhau
        
        Returns:
            keras.Model: Multi-branch CNN model
        """
        input_rgb = layers.Input(shape=self.input_shape, name='rgb_input')
        input_hsv = layers.Input(shape=self.input_shape, name='hsv_input')
        
        # Branch 1: RGB CNN
        rgb_branch = self._create_cnn_branch(input_rgb, name='rgb_branch')
        
        # Branch 2: HSV CNN
        hsv_branch = self._create_cnn_branch(input_hsv, name='hsv_branch')
        
        # Concatenate branches
        concatenated = layers.Concatenate()([rgb_branch, hsv_branch])
        
        # Spatial Attention (tập trung vào vùng top của ảnh)
        attention = self._create_spatial_attention(concatenated)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(attention)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Output
        output = layers.Dense(2, activation='softmax', name='output')(x)
        
        model = keras.Model(inputs=[input_rgb, input_hsv], outputs=output)
        
        return model
    
    def _create_cnn_branch(self, input_layer, name=''):
        """Tạo một CNN branch"""
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
        
        x = layers.GlobalAveragePooling2D(name=f'{name}_gap')(x)
        
        return x
    
    def _create_spatial_attention(self, input_tensor):
        """
        Tạo spatial attention mechanism
        Tập trung vào vùng quan trọng (vùng đầu/mũ bảo hiểm)
        """
        # Tính attention weights
        attention = layers.Dense(256, activation='tanh', name='attention_dense1')(input_tensor)
        attention = layers.Dense(128, activation='sigmoid', name='attention_dense2')(attention)
        
        # Apply attention
        attended = layers.Multiply(name='attention_apply')([input_tensor, attention])
        
        return attended
    
    def preprocess_for_rgb_hsv(self, image):
        """
        Preprocess ảnh cho multi-branch model (RGB và HSV)
        
        Args:
            image: BGR image từ OpenCV
            
        Returns:
            tuple: (rgb_image, hsv_image) đã normalized
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_shape[0], self.input_shape[1]))
        rgb = rgb.astype(np.float32) / 255.0
        
        # Convert to HSV và normalize
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = cv2.resize(hsv, (self.input_shape[0], self.input_shape[1]))
        hsv = hsv.astype(np.float32)
        # Normalize HSV: H [0, 360] -> [0, 1], S [0, 255] -> [0, 1], V [0, 255] -> [0, 1]
        hsv[:, :, 0] = hsv[:, :, 0] / 180.0  # H
        hsv[:, :, 1] = hsv[:, :, 1] / 255.0  # S
        hsv[:, :, 2] = hsv[:, :, 2] / 255.0  # V
        
        return rgb, hsv


class ImprovedHelmetModel:
    """Model cải tiến với nhiều loại features"""
    
    def __init__(self, input_shape=(224, 224, 3)):
        self.feature_extractor = ImprovedHelmetFeatureExtractor(input_shape)
        self.model = None
    
    def create_hybrid_model(self, use_handcrafted=True):
        """
        Tạo hybrid model kết hợp CNN và handcrafted features
        
        Args:
            use_handcrafted: Có sử dụng handcrafted features không
        """
        if use_handcrafted:
            # Model với handcrafted features
            # ... (cần implement riêng)
            pass
        
        # Model với multi-branch CNN
        self.model = self.feature_extractor.create_multi_branch_cnn()
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model"""
        if self.model is None:
            raise ValueError("Model chưa được tạo!")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def predict_with_preprocessing(self, image):
        """
        Dự đoán với preprocessing RGB và HSV
        
        Args:
            image: BGR image từ OpenCV
            
        Returns:
            tuple: (prediction, confidence)
        """
        if self.model is None:
            raise ValueError("Model chưa được load!")
        
        # Preprocess
        rgb, hsv = self.feature_extractor.preprocess_for_rgb_hsv(image)
        
        # Predict
        rgb_batch = np.expand_dims(rgb, axis=0)
        hsv_batch = np.expand_dims(hsv, axis=0)
        
        prediction = self.model.predict([rgb_batch, hsv_batch], verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        class_names = ["Không có mũ", "Có mũ bảo hiểm"]
        return class_names[predicted_class], confidence


# Example usage
if __name__ == "__main__":
    print("=== IMPROVED FEATURE EXTRACTION ===")
    
    # Test feature extraction
    extractor = ImprovedHelmetFeatureExtractor()
    
    # Tạo ảnh test (random)
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Extract features
    print("\n1. Extracting color features...")
    color_features = extractor.extract_color_features(test_image)
    print(f"   Color features: {len(color_features)} types")
    
    print("\n2. Extracting texture features...")
    texture_features = extractor.extract_texture_features(test_image)
    print(f"   Texture features: {len(texture_features)} types")
    
    print("\n3. Extracting spatial features...")
    spatial_features = extractor.extract_spatial_features(test_image)
    print(f"   Spatial features: {len(spatial_features)} types")
    
    print("\n4. Extracting all features...")
    all_features = extractor.extract_all_features(test_image)
    print(f"   Total feature vector length: {len(all_features)}")
    
    print("\n✅ Feature extraction completed!")
    
    # Test model creation
    print("\n5. Creating multi-branch CNN model...")
    model = extractor.create_multi_branch_cnn()
    print(f"   Model created with {model.count_params():,} parameters")
    print(f"   Model summary:")
    model.summary()

