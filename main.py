import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import base64
import os
import sys
import torch
import torch.nn as nn
import uuid
import shutil
from typing import List
from glob import glob
from torchvision import models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ultralytics"))
from ultralytics import YOLO

# Monkeypatch torch.load for PyTorch 2.6+ compatibility
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

app = FastAPI()

# Directories
UPLOAD_DIR = "static/uploads"
FEEDBACK_DIR = "feedback_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(f"{FEEDBACK_DIR}/images", exist_ok=True)
os.makedirs(f"{FEEDBACK_DIR}/labels", exist_ok=True)

# Model Paths
def get_latest_model():
    # Prefer latest feedback-tuned model when available.
    all_detect_runs = glob("runs/detect/**/weights/best.pt", recursive=True)
    feedback_runs = [p for p in all_detect_runs if "finetuned_feedback" in p]
    if feedback_runs:
        feedback_runs.sort(key=os.path.getmtime, reverse=True)
        return feedback_runs[0]

    # Otherwise use latest trained detection checkpoint from runs/.
    if all_detect_runs:
        all_detect_runs.sort(key=os.path.getmtime, reverse=True)
        return all_detect_runs[0]

    # Final fallbacks if no trained weights exist yet.
    fallbacks = [
        "weights/best.pt",
        "runs/detect/broken_street_v8x/weights/best.pt",
        "runs/detect/pothole_v8x/weights/best.pt",
        "yolov8m.pt"
    ]
    for p in fallbacks:
        if os.path.exists(p):
            return p
    return "yolov8m.pt"


def get_latest_severity_model():
    preferred = [
        "weights/severity_efficientnet_b0.pth",
        "weights/severity_efficientnet.pth",
    ]
    for p in preferred:
        if os.path.exists(p):
            return p

    cls_runs = glob("runs/classify/**/severity_efficientnet*.pth", recursive=True)
    if cls_runs:
        cls_runs.sort(key=os.path.getmtime, reverse=True)
        return cls_runs[0]

    weights_runs = glob("weights/severity_efficientnet*.pth", recursive=False)
    if weights_runs:
        weights_runs.sort(key=os.path.getmtime, reverse=True)
        return weights_runs[0]
    return None


def load_efficientnet_severity(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names", ["minor_pothole", "medium_pothole", "major_pothole"])
    arch = checkpoint.get("arch", "efficientnet_b0")
    if arch != "efficientnet_b0":
        raise ValueError(f"Unsupported EfficientNet arch: {arch}")

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return model, class_names, device, mean, std


def predict_severity_efficientnet(crop_bgr, model, class_names, device, mean, std):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)
    x = (x - mean) / std

    with torch.no_grad():
        logits = model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())
    return class_names[pred_idx]

DETECTION_MODEL_PATH = get_latest_model()
print(f"Using Detection Model: {DETECTION_MODEL_PATH}")

SEVERITY_MODEL_PATH = get_latest_severity_model()
print(f"Using Severity Model: {SEVERITY_MODEL_PATH if SEVERITY_MODEL_PATH else 'None'}")

detection_model = YOLO(DETECTION_MODEL_PATH)
severity_model = None
severity_class_names = None
severity_device = None
severity_mean = None
severity_std = None

if SEVERITY_MODEL_PATH:
    severity_model, severity_class_names, severity_device, severity_mean, severity_std = load_efficientnet_severity(SEVERITY_MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return f.read()
    return "<h1>API is running</h1>"

def img_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def _is_severity_candidate(class_name: str) -> bool:
    name = class_name.lower().strip()
    keywords = ["pothole", "broken", "crack", "road", "defect"]
    return any(k in name for k in keywords)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save original image for potential feedback
    image_id = str(uuid.uuid4())
    img_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    cv2.imwrite(img_path, img)
    
    # 1. M1 Detection
    results = detection_model(img)
    
    m1_img = img.copy()
    m2_img = img.copy()
    
    counts = {"total": 0, "minor": 0, "medium": 0, "major": 0}
    detections = []
    
    for result in results:
        for box in result.boxes:
            b = box.xyxy[0].tolist() # x1, y1, x2, y2
            conf = float(box.conf)
            cls_id = int(box.cls)
            det_class = result.names.get(cls_id, str(cls_id))
            counts["total"] += 1
            
            # 2. M2 Severity
            severity = "unknown"
            if severity_model and _is_severity_candidate(det_class):
                x1, y1, x2, y2 = map(int, b)
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    severity = predict_severity_efficientnet(
                        crop,
                        severity_model,
                        severity_class_names,
                        severity_device,
                        severity_mean,
                        severity_std,
                    )
            
            if "minor" in severity: counts["minor"] += 1
            elif "medium" in severity: counts["medium"] += 1
            elif "major" in severity: counts["major"] += 1
            
            detections.append({
                "box": b,
                "conf": conf,
                "class_name": det_class,
                "severity": severity
            })

            # Draw M1 (Simple Blue Box)
            cv2.rectangle(m1_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
            cv2.putText(m1_img, f"{det_class} {conf:.2f}", (int(b[0]), int(b[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw M2 (Severity Coded)
            color = (0, 255, 0) # Green
            if "major" in severity: color = (0, 0, 255) # Red
            elif "medium" in severity: color = (0, 165, 255) # Orange
            
            cv2.rectangle(m2_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 3)
            m2_label = severity.split('_')[0].upper() if severity != "unknown" else det_class.upper()
            cv2.putText(m2_img, m2_label, (int(b[0]), int(b[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return JSONResponse(content={
        "image_id": image_id,
        "m1_base64": img_to_base64(m1_img),
        "m2_base64": img_to_base64(m2_img),
        "counts": counts,
        "detections": detections,
        "image_size": [img.shape[1], img.shape[0]] # width, height
    })

@app.post("/feedback")
async def feedback(image_id: str = Form(...), is_false_positive: bool = Form(...)):
    # Simple feedback: If flagged as false positive, save as hard negative
    src_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    if not os.path.exists(src_path):
        return JSONResponse(content={"status": "error", "message": "Image not found"}, status_code=404)
    
    if is_false_positive:
        dest_img_path = os.path.join(FEEDBACK_DIR, "images", f"{image_id}.jpg")
        dest_lbl_path = os.path.join(FEEDBACK_DIR, "labels", f"{image_id}.txt")
        
        # Move image to feedback folder
        shutil.move(src_path, dest_img_path)
        
        # Create empty label file (Hard Negative)
        with open(dest_lbl_path, "w") as f:
            pass
            
        return JSONResponse(content={"status": "success", "message": "Feedback saved as hard negative"})
    
    return JSONResponse(content={"status": "ignored"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
