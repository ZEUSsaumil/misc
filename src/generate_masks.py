import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import glob

# Label mapping
PART_LABELS = {"spare part outline", "part", "good", "good_part"}
DEFECT_LABEL_MAP = {
    "scratch": 2,
    "dent": 3,
    "chip": 4,
    "defect": 2,
}


def _label_to_class_id(label: str) -> int:
    lbl = label.strip().lower()
    if lbl in PART_LABELS:
        return 1
    if lbl in DEFECT_LABEL_MAP:
        return DEFECT_LABEL_MAP[lbl]
    for key, cid in DEFECT_LABEL_MAP.items():
        if key in lbl:
            return cid
    return 2  # default defect id

def generate_masks():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_json_dir = os.path.join(base_dir, 'raw dataset', '20251201182754114464_-1', 'LABELED TEXTURE FILES')
    
    # Target directories
    train_img_dir = os.path.join(base_dir, 'data', 'train', 'images')
    val_img_dir = os.path.join(base_dir, 'data', 'val', 'images')
    
    train_mask_dir = os.path.join(base_dir, 'data', 'train', 'masks_multiclass')
    val_mask_dir = os.path.join(base_dir, 'data', 'val', 'masks_multiclass')
    
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(raw_json_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files.")
    
    # Map filenames to JSON paths for easy lookup
    # Filename in images: 20251201_..._CAM_Texture.jpg
    # Filename in JSON: 20251201_..._CAM_Texture.json
    json_map = {os.path.basename(f).replace('.json', ''): f for f in json_files}
    
    # Process Train
    print("Processing Training Set...")
    process_set(train_img_dir, train_mask_dir, json_map)
    
    # Process Val
    print("Processing Validation Set...")
    process_set(val_img_dir, val_mask_dir, json_map)

def process_set(img_dir, mask_dir, json_map):
    # Get list of texture images in the directory
    img_files = [f for f in os.listdir(img_dir) if '_CAM_Texture.jpg' in f]
    
    for img_file in tqdm(img_files):
        base_name = img_file.replace('.jpg', '')
        
        if base_name not in json_map:
            print(f"Warning: No JSON found for {img_file}")
            continue
            
        json_path = json_map[base_name]
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Determine image size (assuming all are same or reading from json)
        # Usually labelme json has imageHeight and imageWidth
        h = data.get('imageHeight')
        w = data.get('imageWidth')
        
        # If dimensions missing, read the image to get them
        if h is None or w is None:
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
        # Create blank mask
        # 0: Background
        mask = np.zeros((h, w), dtype=np.uint8)

        shapes = data.get('shapes', [])

        # First pass: draw part (class 1)
        for shape in shapes:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.int32)
            cls_id = _label_to_class_id(label)
            if cls_id == 1:
                cv2.fillPoly(mask, [points], 1)

        # Second pass: draw defects to overwrite part where overlapping
        for shape in shapes:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.int32)
            cls_id = _label_to_class_id(label)
            if cls_id != 1:
                cv2.fillPoly(mask, [points], cls_id)
        
        # Save mask
        # Name convention: ..._CAM_Texture_mask.png
        mask_filename = base_name + "_mask.png"
        cv2.imwrite(os.path.join(mask_dir, mask_filename), mask)

        # Quick check: warn if only background present
        unique_vals = np.unique(mask)
        if unique_vals.size <= 1:
            print(f"Warning: mask {mask_filename} has only values {unique_vals}")

if __name__ == "__main__":
    generate_masks()
