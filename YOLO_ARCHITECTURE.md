# YOLOv8 Model Architecture

The project primarily utilizes the **YOLOv8** (You Only Look Once version 8) architecture, specifically the `yolov8x` (Extra Large) and `yolov8m` (Medium) variants from the Ultralytics library.

## Overview
YOLOv8 is a state-of-the-art, real-time object detection model that improves upon previous YOLO versions by introducing a new backbone network, a new anchor-free detection head, and a new loss function.

### 1. Backbone
The backbone is responsible for feature extraction from the input image. In YOLOv8, it uses a modified version of the **CSPDarknet53** architecture.
- **Stem**: Initial convolution layer with a stride of 2.
- **C2f Modules**: Replaces the C3 modules found in YOLOv5. The "Cross-Stage Partial Network with 2 Convolutions" (C2f) module combines high-level features with contextual information to improve detection accuracy.
- **SPPF (Spatial Pyramid Pooling - Fast)**: Used at the end of the backbone to pool features at different scales, ensuring the model can handle objects of various sizes.

### 2. Neck
The neck sits between the backbone and the head, performing feature fusion.
- **PAN-FPN (Path Aggregation Network)**: YOLOv8 uses a PAN-FPN structure to facilitate the flow of both low-level and high-level information.
- It helps in better localization and feature representation by merging features from different stages of the backbone.

### 3. Head (Detection)
The detection head is where the final predictions (bounding boxes and classes) are made.
- **Decoupled Head**: Unlike previous versions, YOLOv8 uses a decoupled head where classification and localization (regression) are handled by separate branches. This improves convergence and performance.
- **Anchor-Free**: YOLOv8 is an anchor-free model. It predicts the center of an object directly instead of the offset from a predefined anchor box, which reduces the complexity of the box decoding process.
- **Loss Functions**: 
  - **VFL (Varifocal Loss)** for classification.
  - **CIOU (Complete IoU) + DFL (Distribution Focal Loss)** for bounding box regression.

## Specific Variants Used in this Project
- **Detection Model (`yolov8x.pt` / `yolov8m.pt`)**: Used for detecting potholes and broken street lights.
- **Classification Model (`yolov8x-cls.pt`)**: Used for pothole severity classification (Minor, Medium, Major).

## Summary Table
| Component | Implementation |
| :--- | :--- |
| **Backbone** | CSPDarknet with C2f Modules |
| **Neck** | PAN-FPN |
| **Head** | Decoupled, Anchor-Free |
| **Activation** | SiLU (Sigmoid Linear Unit) |
| **Input Size** | 640x640 (Detection), 224x224 (Classification) |
