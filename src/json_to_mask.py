import os
import json
import numpy as np
import cv2
import glob
from tqdm import tqdm
import argparse

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
    return 2

def json_to_mask(dataset_dir):
    """
    Converts LabelMe JSON annotations to multiclass mask images.
    Class ids: 0 background, 1 part, defects 2/3/4 as mapped.
    """
    # Find all JSON files
    json_files = glob.glob(os.path.join(dataset_dir, "**", "*.json"), recursive=True)
    
    print(f"Found {len(json_files)} JSON files in {dataset_dir}")
    
    if len(json_files) == 0:
        print("No JSON files found. Please label some images with LabelMe first.")
        return

    count = 0
    for json_file in tqdm(json_files):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # Get image dimensions
            # We try to read the image to be sure of dimensions, or trust the JSON
            image_path = os.path.join(os.path.dirname(json_file), data["imagePath"])
            
            # Fallback: Check if image is in the same directory as JSON
            if not os.path.exists(image_path):
                # Try local path
                local_image_path = os.path.join(os.path.dirname(json_file), os.path.basename(data["imagePath"]))
                
                # Try ../rAW/ path
                raw_image_path = os.path.join(os.path.dirname(json_file), "..", "rAW", os.path.basename(data["imagePath"]))
                
                if os.path.exists(local_image_path):
                    image_path = local_image_path
                    # print(f"Found image at local path: {image_path}")
                elif os.path.exists(raw_image_path):
                    image_path = raw_image_path
                    # print(f"Found image at rAW path: {image_path}")
                else:
                    print(f"Could not find image at {image_path}, {local_image_path}, or {raw_image_path}")

            h = data.get("imageHeight")
            w = data.get("imageWidth")
            
            if h is None or w is None:
                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Warning: Could not read image {image_path}")
                        continue
                    h, w = img.shape[:2]
                else:
                    print(f"Warning: Image {image_path} not found and dimensions not in JSON.")
                    continue
                
            # Create empty mask (black background)
            mask = np.zeros((h, w), dtype=np.uint8)

            # Draw polygons
            shapes = data.get("shapes", [])
            if not shapes:
                print(f"Warning: No shapes found in {json_file}")

            # First pass: part
            for shape in shapes:
                points = np.array(shape["points"], dtype=np.int32)
                cls_id = _label_to_class_id(shape["label"])
                if cls_id == 1:
                    cv2.fillPoly(mask, [points], cls_id)

            # Second pass: defects overwrite
            for shape in shapes:
                points = np.array(shape["points"], dtype=np.int32)
                cls_id = _label_to_class_id(shape["label"])
                if cls_id != 1:
                    cv2.fillPoly(mask, [points], cls_id)
                
            # Save mask
            # Naming convention: if image is "image.jpg", mask is "image_mask.png"
            # Save mask in the same directory as the IMAGE, not the JSON
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(os.path.dirname(image_path), mask_filename)
            
            cv2.imwrite(mask_path, mask)
            count += 1

            uniq = np.unique(mask)
            if uniq.size <= 1:
                print(f"Warning: Mask {mask_filename} has only values {uniq}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"Successfully generated {count} masks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=r"C:\Users\800291\Desktop\SOMIC WORK\somic_supervised_model_2D\20251201182754114464_-1", help="Path to dataset directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory {args.dataset_dir} does not exist.")
    else:
        json_to_mask(args.dataset_dir)
