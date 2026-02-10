import os
import glob
import cv2
import random
import shutil
import xml.etree.ElementTree as ET

# Configuration
INPUT_IMG_DIR = "archive2/images"
INPUT_ANN_DIR = "archive2/annotations"
OUTPUT_DIR = "severity_dataset"
CLASSES = ["minor_pothole", "medium_pothole", "major_pothole"]
TRAIN_SPLIT = 0.8

def setup_directories():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

def process_dataset():
    print("Cropping potholes for severity classification...")
    xml_files = glob.glob(os.path.join(INPUT_ANN_DIR, "*.xml"))
    
    count = 0
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            filename = root.find("filename").text
            img_path = os.path.join(INPUT_IMG_DIR, filename)
            
            if not os.path.exists(img_path):
                # Try with common extensions if exact filename fails
                name_no_ext = os.path.splitext(os.path.basename(xml_file))[0]
                potential = glob.glob(os.path.join(INPUT_IMG_DIR, name_no_ext + ".*"))
                if potential:
                    img_path = potential[0]
                else:
                    continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_h, img_w = img.shape[:2]
            
            for i, obj in enumerate(root.findall("object")):
                name = obj.find("name").text
                if name not in CLASSES:
                    continue
                
                bndbox = obj.find("bndbox")
                xmin = max(0, int(float(bndbox.find("xmin").text)))
                ymin = max(0, int(float(bndbox.find("ymin").text)))
                xmax = min(img_w, int(float(bndbox.find("xmax").text)))
                ymax = min(img_h, int(float(bndbox.find("ymax").text)))
                
                if xmax <= xmin or ymax <= ymin:
                    continue
                    
                # Crop
                crop = img[ymin:ymax, xmin:xmax]
                
                # Determine split
                split = "train" if random.random() < TRAIN_SPLIT else "val"
                
                # Save crop
                crop_name = f"{os.path.splitext(filename)[0]}_obj{i}.jpg"
                dest_path = os.path.join(OUTPUT_DIR, split, name, crop_name)
                cv2.imwrite(dest_path, crop)
                count += 1
                
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            
    print(f"Finished processing. Total crops saved: {count}")

if __name__ == "__main__":
    setup_directories()
    process_dataset()
