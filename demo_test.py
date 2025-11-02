import cv2
import numpy as np
from ultralytics import YOLO
import os

def create_demo_image():
    """Tạo ảnh demo để test"""
    # Tạo ảnh trắng
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Vẽ một số hình đơn giản để test
    cv2.rectangle(img, (100, 100), (200, 300), (0, 0, 0), 2)  # Hình chữ nhật
    cv2.circle(img, (400, 200), 50, (0, 0, 0), 2)  # Hình tròn
    
    # Thêm text
    cv2.putText(img, "Demo Image for Testing", (150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Lưu ảnh
    cv2.imwrite("demo_image.jpg", img)
    print("Đã tạo ảnh demo: demo_image.jpg")

def test_yolo_model():
    """Test model YOLO"""
    try:
        print("Đang tải model YOLO...")
        model = YOLO('yolov8n.pt')
        print("Model đã được tải thành công!")
        
        # Test với ảnh demo
        if os.path.exists("demo_image.jpg"):
            print("Đang test với ảnh demo...")
            results = model("demo_image.jpg")
            print(f"Số lượng object được phát hiện: {len(results[0].boxes) if results[0].boxes is not None else 0}")
        
        return True
    except Exception as e:
        print(f"Lỗi khi test model: {e}")
        return False

def main():
    print("=== DEMO VÀ TEST ỨNG DỤNG NHẬN DIỆN MẶT NGƯỜI ===")
    print()
    
    # Tạo ảnh demo
    create_demo_image()
    
    # Test model YOLO
    if test_yolo_model():
        print()
        print("✅ Tất cả test đều thành công!")
        print("Bạn có thể chạy ứng dụng chính bằng:")
        print("  - Windows: run_app.bat")
        print("  - Linux/Mac: ./run_app.sh")
        print("  - Hoặc: python face_detection_app.py")
    else:
        print()
        print("❌ Có lỗi xảy ra. Vui lòng kiểm tra lại cài đặt.")

if __name__ == "__main__":
    main()
