import os
import json
import shutil
import random
import glob
import xml.etree.ElementTree as ET

# Configuration
ARCHIVE2_DIR = "archive2"
NINJA_DIR = "pothole-detection-DatasetNinja/ds"
ROBO_DIR = "Street-Light-1"
OUTPUT_DIR = "yolo_dataset"

# Final Classes
# 0: broken street
# 1: street light
CLASSES = ["broken street", "street light"]

# Mapping source labels to target indices
MAPPING = {
    "pothole": 0,
    "minor_pothole": 0,
    "medium_pothole": 0,
    "major_pothole": 0,
    "street_light_0": 1,
    "street light": 1  # in case it appears differently
}

def setup_directories():
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

def convert_to_yolo(box, img_w, img_h, class_idx):
    xmin, ymin, xmax, ymax = box
    xmin = max(0, min(xmin, img_w))
    ymin = max(0, min(ymin, img_h))
    xmax = max(0, min(xmax, img_w))
    ymax = max(0, min(ymax, img_h))
    
    if xmax <= xmin or ymax <= ymin:
        return None
        
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    
    return f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_archive2_dataset():
    print("Processing Archive2 Dataset (Broken Street)...")
    splits_path = os.path.join("archive", "splits.json")
    val_files = set()
    train_files = set()
    if os.path.exists(splits_path):
        with open(splits_path, 'r') as f:
            splits = json.load(f)
            val_files = set(splits.get("test", []))
            train_files = set(splits.get("train", []))
    
    xml_files = glob.glob(os.path.join(ARCHIVE2_DIR, "annotations", "*.xml"))
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            xml_basename = os.path.basename(xml_file)
            
            if xml_basename in val_files: split = "val"
            elif xml_basename in train_files: split = "train"
            else: split = "train" if random.random() < 0.8 else "val"
            
            filename = root.find("filename").text if root.find("filename") is not None else xml_basename.replace(".xml", ".jpg")
            img_path = os.path.join(ARCHIVE2_DIR, "images", filename)
            if not os.path.exists(img_path):
                name_no_ext = os.path.splitext(xml_basename)[0]
                potential = glob.glob(os.path.join(ARCHIVE2_DIR, "images", name_no_ext + ".*"))
                if potential: img_path = potential[0]
                else: continue
            
            size = root.find("size")
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)
            
            yolo_lines = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name in MAPPING and MAPPING[name] == 0:
                    bndbox = obj.find("bndbox")
                    box = [float(bndbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                    line = convert_to_yolo(box, img_w, img_h, 0)
                    if line: yolo_lines.append(line)
            
            if yolo_lines:
                dest_name = f"archive2_{os.path.basename(img_path)}"
                shutil.copy(img_path, os.path.join(OUTPUT_DIR, "images", split, dest_name))
                with open(os.path.join(OUTPUT_DIR, "labels", split, os.path.splitext(dest_name)[0] + ".txt"), "w") as f:
                    f.write("\n".join(yolo_lines))
        except Exception as e: print(f"Error archive2 {xml_file}: {e}")

def process_ninja_dataset():
    print("Processing DatasetNinja (Broken Street)...")
    ann_files = glob.glob(os.path.join(NINJA_DIR, "ann", "*.json"))
    for ann_file in ann_files:
        try:
            with open(ann_file, 'r') as f: data = json.load(f)
            img_h, img_w = data["size"]["height"], data["size"]["width"]
            img_filename = os.path.basename(ann_file).replace(".json", "")
            img_path = os.path.join(NINJA_DIR, "img", img_filename)
            if not os.path.exists(img_path): continue
                 
            yolo_lines = []
            for obj in data["objects"]:
                if obj["classTitle"] in MAPPING and MAPPING[obj["classTitle"]] == 0:
                    pts = obj["points"]["exterior"]
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    line = convert_to_yolo([min(xs), min(ys), max(xs), max(ys)], img_w, img_h, 0)
                    if line: yolo_lines.append(line)
            
            if yolo_lines:
                split = "train" if random.random() < 0.8 else "val"
                dest_name = f"ninja_{img_filename}"
                shutil.copy(img_path, os.path.join(OUTPUT_DIR, "images", split, dest_name))
                with open(os.path.join(OUTPUT_DIR, "labels", split, os.path.splitext(dest_name)[0] + ".txt"), "w") as f:
                    f.write("\n".join(yolo_lines))
        except Exception as e: print(f"Error ninja {ann_file}: {e}")

def process_roboflow_dataset():
    print("Processing Roboflow Dataset (Street Light)...")
    # Street-Light-1 has train, valid, test. We map valid and test to val.
    for robo_split, target_split in [("train", "train"), ("valid", "val"), ("test", "val")]:
        img_dir = os.path.join(ROBO_DIR, robo_split, "images")
        lbl_dir = os.path.join(ROBO_DIR, robo_split, "labels")
        
        if not os.path.exists(img_dir): continue
        
        files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        for lbl_file in files:
            img_base = os.path.splitext(os.path.basename(lbl_file))[0]
            # Try extensions
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = os.path.join(img_dir, img_base + ext)
                if os.path.exists(p):
                    img_path = p
                    break
            
            if not img_path: continue
            
            # Read and remap class index to 1 (street light)
            new_lines = []
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    # Roboflow usually has class 0, but we MUST ensure it's mapped to 1
                    # Since it's a single class dataset, all annotations are street lights
                    parts[0] = "1"
                    new_lines.append(" ".join(parts))
            
            if new_lines:
                dest_name = f"robo_{os.path.basename(img_path)}"
                shutil.copy(img_path, os.path.join(OUTPUT_DIR, "images", target_split, dest_name))
                with open(os.path.join(OUTPUT_DIR, "labels", target_split, os.path.splitext(dest_name)[0] + ".txt"), "w") as f:
                    f.write("\n".join(new_lines))

def update_data_yaml():
    print("Updating data.yaml...")
    content = """path: yolo_dataset
train: images/train
val: images/val

names:
  0: broken street
  1: street light
"""
    with open("data.yaml", "w") as f: f.write(content)

if __name__ == "__main__":
    setup_directories()
    process_archive2_dataset()
    process_ninja_dataset()
    process_roboflow_dataset()
    update_data_yaml()
    print("Dataset preparation complete.")