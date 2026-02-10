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
        batch=8,       # Adjusted for 8GB VRAM
        name="broken_street_v8x",
        project="runs/detect"
    )
    
    print("Training complete.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    train()
