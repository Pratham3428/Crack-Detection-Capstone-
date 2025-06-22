import os
import argparse
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import glob
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Test YOLOv12 Road Cracks Detection Model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True, help='Path to image, video, or directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto, cuda:0, etc.)')
    parser.add_argument('--save-dir', type=str, default='road_cracks_results/detections', help='Directory to save results')
    parser.add_argument('--view', action='store_true', help='Display results in real-time')
    return parser.parse_args()

def process_image(model, image_path, args, save_dir):
    """Process a single image and save/display the results"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Run inference
    results = model.predict(
        source=img,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.img_size,
        device=args.device,
        verbose=False
    )
    
    # Process results
    result = results[0]
    annotated_img = result.plot()
    
    # Get class names
    class_names = model.names
    
    # Add text with detection information
    boxes = result.boxes
    for i, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        class_name = class_names[cls_id]
        print(f"Detected {class_name} with confidence {conf:.2f}")
    
    # Save the result
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, annotated_img)
    print(f"Saved result to {save_path}")
    
    # Display if requested
    if args.view:
        cv2.imshow('Detection Result', annotated_img)
        cv2.waitKey(0)
    
    return annotated_img

def process_video(model, video_path, args, save_dir):
    """Process a video and save/display the results"""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer
    save_path = os.path.join(save_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.img_size,
            device=args.device,
            verbose=False
        )
        
        # Process results
        result = results[0]
        annotated_frame = result.plot()
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Display if requested
        if args.view:
            cv2.imshow('Detection Result', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    if args.view:
        cv2.destroyAllWindows()
    
    print(f"Saved result to {save_path}")

def main():
    args = parse_args()
    
    # Check if CUDA is available
    if not args.device:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Running inference on device: {args.device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    try:
        model = YOLO(args.model)
        print(f"Loaded model from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process source
    source_path = args.source
    if os.path.isfile(source_path):
        # Check if it's an image or video
        if source_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            process_image(model, source_path, args, args.save_dir)
        elif source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            process_video(model, source_path, args, args.save_dir)
        else:
            print(f"Unsupported file format: {source_path}")
    elif os.path.isdir(source_path):
        # Process all images in directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(source_path, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(source_path, f'*{ext.upper()}')))
        
        if not image_files:
            print(f"No image files found in {source_path}")
            return
        
        print(f"Found {len(image_files)} images to process")
        for img_path in image_files:
            process_image(model, img_path, args, args.save_dir)
    else:
        print(f"Source not found: {source_path}")

if __name__ == "__main__":
    main()
