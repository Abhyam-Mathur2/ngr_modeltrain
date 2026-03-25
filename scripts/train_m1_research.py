import torch
from ultralytics import YOLO

def train_m1_research():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = '0'
    else:
        print("WARNING: CUDA is not available. Training will fall back to CPU and will be very slow.")
        device = 'cpu'

    # Load the new custom research architecture
    # We do not load pretrained weights (like yolov8m.pt) because our architecture is modified (C2fCoordAtt)
    # The weights wouldn't align. We are training from scratch.
    print("Initializing YOLOv8 with Research Architecture (CoordAtt)...")
    model = YOLO('yolov8-pothole-research.yaml') 

    print("Starting Training with Lion Optimizer and MPDIoU...")
    # 'data.yaml' is your existing dataset. 
    # The YAML indicates nc: 1 (pothole only), so it will only train on the first class.
    results = model.train(
        data='../data.yaml',
        epochs=50, # Adjust as needed
        imgsz=640,
        batch=16,
        device=device,
        optimizer='Lion', # Using our new Lion optimizer
        lr0=0.0001,       # Lion requires a smaller learning rate than AdamW
        box=7.5,          # Box loss weight. Since we use MPDIoU, this helps focus on precise corners
        project='runs/detect',
        name='research_m1_lion_mpdiou',
        exist_ok=True,
        workers=4,        # Number of dataloader workers
        activation='Mish' # Specify our new Mish activation
    )
    print("Research M1 Training complete.")

if __name__ == "__main__":
    train_m1_research()
