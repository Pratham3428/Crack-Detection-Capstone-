import os
import torch
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv12 on Road Cracks Dataset')
    parser.add_argument('--model', type=str, default='yolov12n.pt', 
                        choices=['yolov12n.pt', 'yolov12s.pt', 'yolov12m.pt', 'yolov12l.pt', 'yolov12x.pt'],
                        help='YOLOv12 model size to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto, or cuda:0, etc.)')
    parser.add_argument('--project', type=str, default='road_cracks_results', help='Project name for saving results')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    args = parser.parse_args()

    # Check if CUDA is available
    if not args.device:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Training on device: {args.device}")
    
    # Create a custom data.yaml file with absolute paths
    dataset_dir = os.path.abspath('roadCracksDataset')
    custom_yaml_path = os.path.join(dataset_dir, 'custom_data.yaml')
    
    with open(os.path.join(dataset_dir, 'data.yaml'), 'r') as f:
        data_yaml = f.read()
    
    # Update paths to be absolute
    with open(custom_yaml_path, 'w') as f:
        f.write(f"train: {os.path.join(dataset_dir, 'train/images')}\n")
        f.write(f"val: {os.path.join(dataset_dir, 'valid/images')}\n")
        f.write(f"test: {os.path.join(dataset_dir, 'test/images')}\n\n")
        f.write("names:\n")
        f.write("  0: alligator crack\n")
        f.write("  1: lateral crack\n")
        f.write("  2: longitudinal crack\n")
        f.write("  3: pothole\n")
    
    print(f"Created custom data YAML at: {custom_yaml_path}")
    
    # Load the model
    model = YOLO(args.model)
    print(f"Loaded model: {args.model}")
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    model.train(
        data=custom_yaml_path,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=True,
        patience=50,  # Early stopping patience
        save=True,    # Save checkpoints
        save_period=10,  # Save every 10 epochs
        val=True      # Run validation
    )
    
    # Evaluate the model on the test set
    print("Evaluating model on test set...")
    metrics = model.val(data=custom_yaml_path, split='test')
    print(f"Test set metrics: {metrics}")
    
    # Save the trained model
    final_model_path = os.path.join(args.project, args.name, 'weights', 'best.pt')
    print(f"Best model saved at: {final_model_path}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
