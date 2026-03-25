import torch
from ultralytics import YOLO
import os

# Monkeypatch torch.load for compatibility
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

def train():
    # Path to last checkpoint
    last_checkpoint = "runs/detect/runs/detect/refined_pothole_v8m/weights/last.pt"
    
    if os.path.exists(last_checkpoint):
        print(f"Resuming training from {last_checkpoint}")
        model = YOLO(last_checkpoint)
        results = model.train(resume=True, workers=4)
    else:
        # Load YOLOv8 medium model
        model = YOLO("yolov8m.pt")

        # Train the model
        # Using refined_data.yaml which contains potholes + good road background
        results = model.train(
            data="refined_data.yaml",
            epochs=50,
            imgsz=640,
            device=0,      # Use GPU 0
            batch=8,       # Medium model is heavier than Nano, 8 is usually safe for 8GB+ VRAM
            name="refined_pothole_v8m",
            project="runs/detect",
            
            # Explicit Optimization and Loss Configuration
            optimizer='AdamW',  # Optimization method
            lr0=0.01,           # Initial learning rate
            box=7.5,            # Box loss gain
            cls=0.5,            # Class loss gain
            dfl=1.5             # Distribution Focal Loss gain
        )
    
    print("Refined Training complete.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    train()
