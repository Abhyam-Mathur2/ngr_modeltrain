# M1/M2 Pothole Detection & Severity Improvement Plan

This plan focuses on optimizing the dual-model pipeline: **M1 (Detection)** and **M2 (Severity Analysis)**.

## 1. M1: Detection Optimization (The "Where" & "How Many")

### 1.1 Model Selection & Tuning
*   **Ensemble Detection:** Combine your current YOLOv8m with a smaller, faster YOLOv11n or YOLOv10n. Only confirm a pothole if both models overlap (reduces false positives) or if either detects it with high confidence (increases recall).
*   **Focus on Box Loss:** Switch from standard **CIoU** to **Wise-IoU (WIoU)**. WIoU is specifically designed to handle "outliers" (very small or very large potholes) better than standard IoU.
*   **SAHI (Slicing Aided Hyper Inference):** If your camera is high-resolution, potholes often appear small. SAHI slices the image into patches, detects potholes in each, and merges them, significantly boosting detection of tiny cracks.

### 1.2 Training Data Enhancements
*   **Background Class (Null Labels):** Ensure 10-15% of your training data consists of "Good Road" images with *no* labels. This is the single best way to stop the model from hallucinating potholes in shadows.

---

## 2. M2: Severity Optimization (The "How Bad")

### 2.1 From YOLO-cls to Custom CNN? (Your Suggestion)
Currently, you use `yolov8x-cls`. While powerful, it's a "heavy" model designed for general objects. For pothole severity (Minor/Medium/Major), a specialized CNN backbone might be better:

*   **EfficientNetV2-S or MobileNetV3-Large:** These are optimized for feature extraction in specific textures. They are much lighter than YOLOv8x-cls and often better at distinguishing subtle texture differences (e.g., depth of a hole).
*   **Ordinal Regression:** Pothole severity is *ordinal* (Minor < Medium < Major). Standard classifiers treat them as unrelated categories. Using an **Ordinal Loss** function in a custom CNN will penalize the model more for guessing "Minor" when it's "Major" than if it guessed "Medium."

### 2.2 Input Refinement
*   **Contextual Cropping:** Don't just crop the pothole; crop it with a 15% margin of the surrounding road. The contrast between the "good road" and the "hole" is a key feature for severity.
*   **Multi-View Inference:** If M1 detects a pothole, crop it and also crop a "flipped" and "rotated" version. Pass all three to M2 and average the severity scores for a more stable result.

---

## 3. Pipeline Integration (M1 + M2)

### 3.1 The "Filter" Stage
Add a "Certainty" check between M1 and M2. If M1 is less than 40% sure it's a pothole, don't waste power running M2.

### 3.2 Metadata Injection
Pass the **size** of the bounding box from M1 into the final severity calculation. A pothole that takes up 50% of the lane is almost always "Major," regardless of what the classifier thinks.

---

## 4. Immediate Execution Steps

1.  **M1:** Retrain with **WIoU** and **Background Images** (Good Road).
2.  **M2:** Experiment with an **EfficientNetV2** backbone using PyTorch. Compare its accuracy and speed against your current `yolov8x-cls`.
3.  **Deployment:** Update `main.py` to implement the "Metadata Injection" (Box Area + Severity Class).

---
*Updated for ngr-yolov8 dual-model architecture*
