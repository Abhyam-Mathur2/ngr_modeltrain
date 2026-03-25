import os
import shutil
import random

def prepare():
    src_yolo = "yolo_dataset"
    dst_yolo = "refined_dataset"
    
    # Create structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dst_yolo, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dst_yolo, "labels", split), exist_ok=True)

    # 1. Copy original images and filter labels
    for split in ['train', 'val']:
        img_dir = os.path.join(src_yolo, "images", split)
        lbl_dir = os.path.join(src_yolo, "labels", split)
        
        if not os.path.exists(img_dir): continue
        
        for f in os.listdir(img_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(img_dir, f), os.path.join(dst_yolo, "images", split, f))
                
                # Copy and filter label
                label_name = os.path.splitext(f)[0] + ".txt"
                src_label = os.path.join(lbl_dir, label_name)
                dst_label = os.path.join(dst_yolo, "labels", split, label_name)
                
                if os.path.exists(src_label):
                    with open(src_label, 'r') as f_in:
                        lines = f_in.readlines()
                    
                    # Filter: only keep class 0 (broken street)
                    # Mapping: class 0 remains 0. Class 1 (street light) is removed.
                    filtered_lines = [l for l in lines if l.startswith('0 ')]
                    
                    with open(dst_label, 'w') as f_out:
                        f_out.writelines(filtered_lines)
                else:
                    # Create empty label file for background images if they were already there
                    open(dst_label, 'a').close()

    # 2. Add Good Road background images
    good_road_dirs = [
        r"D:\Road_Condition_Dataset\Clean road",
        r"D:\Road_Condition_Dataset\Smooth road"
    ]
    
    all_good_images = []
    for d in good_road_dirs:
        if os.path.exists(d):
            imgs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.jpg', '.jpeg', '.png'))]
            all_good_images.extend(imgs)
    
    random.shuffle(all_good_images)
    
    # Split 80/20 for background
    split_idx = int(len(all_good_images) * 0.8)
    train_good = all_good_images[:split_idx]
    val_good = all_good_images[split_idx:]
    
    for img_path in train_good:
        name = "bg_" + os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dst_yolo, "images", "train", name))
        # No label file needed for background in YOLOv8, but some prefer empty files.
        # We'll leave it without label files as it's standard.
        
    for img_path in val_good:
        name = "bg_" + os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dst_yolo, "images", "val", name))

    print(f"Dataset prepared in {dst_yolo}")
    print(f"Added {len(train_good)} train and {len(val_good)} val background images.")

if __name__ == "__main__":
    prepare()
