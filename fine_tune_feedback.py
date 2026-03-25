import torch
from ultralytics import YOLO
import os
import shutil
from glob import glob

# Monkeypatch torch.load for compatibility
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

FEEDBACK_DIR = "feedback_data"
BASE_DATASET = "refined_dataset"
COMBINED_DATASET = os.path.abspath("combined_dataset") # Use absolute path

def prepare_combined_dataset():
    print(f"Preparing combined dataset at: {COMBINED_DATASET}")
    
    # 1. Create structure
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(COMBINED_DATASET, sub), exist_ok=True)
    
    # 2. Copy original data
    print("Copying base dataset...")
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        src = os.path.join(BASE_DATASET, sub)
        dst = os.path.join(COMBINED_DATASET, sub)
        # Clear dst first to avoid duplicates if re-running
        for f in glob(os.path.join(dst, "*")):
            os.remove(f)
        for f in glob(os.path.join(src, "*")):
            shutil.copy2(f, dst)
            
    # 3. Add feedback data (all feedback goes to training)
    print("Adding feedback data with oversampling (50x)...")
    fb_images = glob(os.path.join(FEEDBACK_DIR, "images", "*.jpg"))
    for img_path in fb_images:
        img_name = os.path.basename(img_path)
        lbl_name = img_name.replace(".jpg", ".txt")
        lbl_path = os.path.join(FEEDBACK_DIR, "labels", lbl_name)
        
        if os.path.exists(lbl_path):
            # Oversample: Copy each feedback image 50 times with unique names
            # This ensures the model sees these "hard negatives" often enough to care
            for i in range(50):
                new_img_name = f"oversample_{i}_{img_name}"
                new_lbl_name = f"oversample_{i}_{lbl_name}"
                shutil.copy2(img_path, os.path.join(COMBINED_DATASET, "images/train", new_img_name))
                shutil.copy2(lbl_path, os.path.join(COMBINED_DATASET, "labels/train", new_lbl_name))
            
    print(f"Combined dataset ready with {len(fb_images)} feedback samples (oversampled to {len(fb_images) * 50}).")

def fine_tune():
    # 1. Prepare data
    prepare_combined_dataset()
    
    # 2. Create YAML with Absolute Path
    yaml_content = f"""
path: {COMBINED_DATASET}
train: images/train
val: images/val

names:
  0: broken street
"""
    yaml_path = os.path.abspath("feedback_combined.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    # 3. Load current best model (Searching common paths)
    possible_paths = [
        "runs/detect/runs/detect/refined_pothole_v8m/weights/best.pt",
        "runs/detect/refined_pothole_v8m/weights/best.pt",
        "weights/best.pt",
        "yolov8m.pt"
    ]
    # Check for existing finetuned models to continue from there
    existing_finetuned = glob("runs/detect/pothole_finetuned_feedback*/weights/best.pt")
    if existing_finetuned:
        existing_finetuned.sort(key=os.path.getmtime, reverse=True)
        possible_paths.insert(0, existing_finetuned[0])

    model_path = "yolov8m.pt"
    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            break
            
    print(f"Loading model for fine-tuning: {model_path}")
    model = YOLO(model_path)
    
    # 4. Train (Fine-tune)
    print("Starting fine-tuning...")
    results = model.train(
        data=yaml_path,
        epochs=30,  # Increased from 10 to 30
        imgsz=640,
        device=0,
        batch=8,
        lr0=0.0005, # Slightly lower learning rate for fine-tuning
        name="pothole_finetuned_feedback",
        project="runs/detect"
    )
    
    print("Fine-tuning complete.")
    print(f"Improved model saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    if not os.path.exists(os.path.join(FEEDBACK_DIR, "images")):
        print("No feedback data found. Collect some feedback in the UI first!")
    else:
        fine_tune()
