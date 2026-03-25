import torch
from ultralytics import YOLO

# Monkeypatch torch.load to default weights_only=False
# This is required for PyTorch 2.6+ when using older versions of ultralytics
# or when loading models that haven't been updated for the new security defaults.
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

def train():
    # Load a model
    # yolov8x is the most accurate (best) detection model in YOLOv8
    # Using detection because the dataset contains only rectangles (boxes)
    model = YOLO("yolov8x.pt")

    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        device=0,      # Use GPU 0
        batch=4,       # Reduced to 4 for yolov8x on 8GB VRAM
        name="broken_street_v8x",
        project="runs/detect",
        
        # Explicit Optimization and Loss Configuration
        optimizer='AdamW',  # Optimization method: AdamW (Adam with Weight Decay)
        lr0=0.01,           # Initial learning rate
        box=7.5,            # Box loss gain (how much the model focuses on box accuracy)
        cls=0.5,            # Class loss gain (how much the model focuses on class accuracy)
        dfl=1.5,            # Distribution Focal Loss gain
        nbs=64              # Nominal batch size for normalization
    )
    
    print("Training complete.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    train()
