import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO
import base64
import os
import torch

# Monkeypatch torch.load for PyTorch 2.6+ compatibility
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

app = FastAPI()

# Model Paths
DETECTION_MODEL_PATH = "runs/detect/runs/detect/pothole_v8x/weights/best.pt"
if not os.path.exists(DETECTION_MODEL_PATH):
    DETECTION_MODEL_PATH = "runs/detect/pothole_v8x/weights/best.pt"
if not os.path.exists(DETECTION_MODEL_PATH):
    DETECTION_MODEL_PATH = "yolov8x.pt"

SEVERITY_MODEL_PATH = "runs/classify/runs/classify/pothole_severity_v8x/weights/best.pt"
if not os.path.exists(SEVERITY_MODEL_PATH):
    SEVERITY_MODEL_PATH = "runs/classify/pothole_severity_v8x/weights/best.pt"

detection_model = YOLO(DETECTION_MODEL_PATH)
severity_model = YOLO(SEVERITY_MODEL_PATH) if os.path.exists(SEVERITY_MODEL_PATH) else None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return f.read()
    return "<h1>API is running</h1>"

def img_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. M1 Detection
    results = detection_model(img)
    
    m1_img = img.copy()
    m2_img = img.copy()
    
    counts = {"total": 0, "minor": 0, "medium": 0, "major": 0}
    
    for result in results:
        for box in result.boxes:
            b = box.xyxy[0].tolist() # x1, y1, x2, y2
            conf = float(box.conf)
            counts["total"] += 1
            
            # Draw M1 (Simple Blue Box)
            cv2.rectangle(m1_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
            cv2.putText(m1_img, f"Pothole {conf:.2f}", (int(b[0]), int(b[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 2. M2 Severity
            severity = "unknown"
            if severity_model:
                x1, y1, x2, y2 = map(int, b)
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    sev_results = severity_model(crop)
                    top1_idx = sev_results[0].probs.top1
                    severity = sev_results[0].names[top1_idx]
            
            if "minor" in severity: counts["minor"] += 1
            elif "medium" in severity: counts["medium"] += 1
            elif "major" in severity: counts["major"] += 1
            
            # Draw M2 (Severity Coded)
            color = (0, 255, 0) # Green
            if "major" in severity: color = (0, 0, 255) # Red
            elif "medium" in severity: color = (0, 165, 255) # Orange
            
            cv2.rectangle(m2_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 3)
            cv2.putText(m2_img, severity.split('_')[0].upper(), (int(b[0]), int(b[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return JSONResponse(content={
        "m1_base64": img_to_base64(m1_img),
        "m2_base64": img_to_base64(m2_img),
        "counts": counts
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
