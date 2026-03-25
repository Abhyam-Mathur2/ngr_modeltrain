import os
import sys

import torch

sys.path.insert(0, "D:/ngr-yolov8/ultralytics")
from ultralytics import YOLO


_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def main():
    ckpt = "D:/ngr-yolov8/runs/detect/runs/detect/broken_street_v8x4/weights/best.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"Starting recovery training from: {ckpt}")
    model = YOLO(ckpt)
    model.train(
        data="data.yaml",
        epochs=12,
        imgsz=512,
        device=0,
        batch=2,
        name="broken_street_v8x4_recover3",
        project="runs/detect",
        workers=0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        optimizer="AdamW",
        lr0=0.01,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        nbs=64,
    )


if __name__ == "__main__":
    main()
