import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch

def test_model(model_name, image_path):
    print(f"\nTesting model: {model_name} on image: {image_path}")
    print(f"Image exists: {os.path.exists(image_path)}")
    
    try:
        # Try to load the model
        print(f"Loading model {model_name}...")
        model = YOLO(model_name)
        print(f"Model loaded successfully: {model}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image {image_path} does not exist")
            return
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        print(f"Image shape: {image.shape}")
        
        # Run inference
        print("Running inference...")
        results = model.predict(source=image, conf=0.25)
        print(f"Inference complete. Results: {results}")
        
        # Check results
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            print(f"Number of detections: {len(boxes)}")
            
            # Print detection details
            for i, box in enumerate(boxes):
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                class_name = model.names[cls_id]
                print(f"Detection {i+1}: {class_name} with confidence {conf:.2f}")
            
            # Save annotated image
            annotated_image = result.plot()
            output_path = f"test_output_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, annotated_image)
            print(f"Saved annotated image to {output_path}")
        else:
            print("No detections found")
    
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    # Test with different models and images
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test with bus image
    bus_image = "ultralytics/assets/bus.jpg"
    test_model("yolov8n.pt", bus_image)  # Test with YOLOv8 (should work)
    test_model("yolov12n.pt", bus_image)  # Test with YOLOv12
    
    # Test with zidane image
    zidane_image = "ultralytics/assets/zidane.jpg"
    test_model("yolov8n.pt", zidane_image)  # Test with YOLOv8 (should work)
    test_model("yolov12n.pt", zidane_image)  # Test with YOLOv12
