import torch
from ultralytics import YOLO
import os

# Monkeypatch torch.load for PyTorch 2.6+ compatibility
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

def train_severity():
    # Using yolov8x-cls.pt for maximum accuracy as requested.
    # It will automatically download if not present.
    model = YOLO("yolov8x-cls.pt")

    # Train the model
    results = model.train(
        data="severity_dataset",
        epochs=50,
        imgsz=224,     # Standard for classification
        device=0,      # Use GPU
        batch=16,      # Adjusted for 8GB VRAM (classification is lighter than detection)
        name="pothole_severity_v8x",
        project="runs/classify"
    )
    
    print("Severity Training complete.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    train_severity()
