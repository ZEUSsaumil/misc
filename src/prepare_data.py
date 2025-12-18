import os
import shutil
import glob
import random
from sklearn.model_selection import train_test_split
from config import Config

def prepare_data(source_dir, split_ratio=0.2):
    """
    Organizes data into train/val folders as expected by Config.
    """
    # Create directories
    for d in [Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, Config.VAL_IMG_DIR, Config.VAL_MASK_DIR]:
        os.makedirs(d, exist_ok=True)
        
    # Find all masks
    mask_files = glob.glob(os.path.join(source_dir, "**", "*_mask.png"), recursive=True)
    
    if not mask_files:
        print(f"No masks found in {source_dir}. Did you run json_to_mask.py?")
        return

    print(f"Found {len(mask_files)} masks.")
    
    # Pair masks with images
    data_pairs = []
    for mask_path in mask_files:
        # Reconstruct image path from mask path
        # mask: name_mask.png -> image: name.jpg
        base_name = os.path.basename(mask_path).replace("_mask.png", "")
        dir_name = os.path.dirname(mask_path)
        
        # Check for jpg, jpeg, png
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            p = os.path.join(dir_name, base_name + ext)
            # print(f"Checking {p}") 
            if os.path.exists(p):
                image_path = p
                break
        
        if image_path:
            data_pairs.append((image_path, mask_path))
        else:
            print(f"Warning: No image found for mask {mask_path}")
            print(f"Expected image at: {os.path.join(dir_name, base_name + '.jpg')}")
            
    if not data_pairs:
        print("No valid image-mask pairs found.")
        return

    # Split
    train_pairs, val_pairs = train_test_split(data_pairs, test_size=split_ratio, random_state=42)
    
    print(f"Training samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    
    def copy_files(pairs, img_dest, mask_dest):
        for img_p, mask_p in pairs:
            shutil.copy2(img_p, img_dest)
            shutil.copy2(mask_p, mask_dest)
            
            # Also copy Normal map if it exists (Required for 6-channel input)
            if "_CAM_Texture" in img_p:
                normal_p = img_p.replace("_CAM_Texture", "_CAM_Normal")
                if os.path.exists(normal_p):
                    shutil.copy2(normal_p, img_dest)
                else:
                    print(f"Warning: Normal map not found for {img_p}")
            
    # Clear existing data? Maybe not, let's just overwrite/add
    # But for clean training, maybe we should clear. 
    # For now, let's just copy.
    
    print("Copying training data...")
    copy_files(train_pairs, Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR)
    
    print("Copying validation data...")
    copy_files(val_pairs, Config.VAL_IMG_DIR, Config.VAL_MASK_DIR)
    
    print("Data preparation complete.")

if __name__ == "__main__":
    # Adjust source dir if needed
    SOURCE_DIR = r"C:\Users\800291\Desktop\SOMIC WORK\somic_supervised_model_2D\raw dataset\20251201182754114464_-1"
    prepare_data(SOURCE_DIR)
