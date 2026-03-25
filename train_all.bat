@echo off
echo Starting YOLOv8 Training (Detection and Severity)...

:: Detection
echo [1/2] Training Detection Model (broken_street_v8x)...
python train.py

:: Severity
echo [2/2] Training Severity Model (pothole_severity_v8x)...
python train_severity.py

echo All training complete!
pause
