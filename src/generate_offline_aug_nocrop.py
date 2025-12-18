import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as albu

from config import Config

# Number of augmented variants per original image
NUM_AUG_PER_IMAGE = 10

# Output folder (clean run using only originals)
OUT_TRAIN_DIR = os.path.join(Config.DATA_DIR, 'train_nocrop_orig_aug')
OUT_IMG_DIR = os.path.join(OUT_TRAIN_DIR, 'images')
OUT_MASK_DIR = os.path.join(OUT_TRAIN_DIR, 'masks_multiclass')


def ensure_dirs():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)


def list_sample_ids(images_dir):
    return sorted([
        f.replace('_CAM_Texture.jpg', '')
        for f in os.listdir(images_dir)
        if f.endswith('_CAM_Texture.jpg')
    ])


def load_triplet(sample_id):
    texture_path = os.path.join(Config.TRAIN_IMG_DIR, f"{sample_id}_CAM_Texture.jpg")
    normal_path = os.path.join(Config.TRAIN_IMG_DIR, f"{sample_id}_CAM_Normal.jpg")
    mask_path = os.path.join(Config.TRAIN_MASK_DIR, f"{sample_id}_CAM_Texture_mask.png")

    texture = cv2.imread(texture_path)
    normal = cv2.imread(normal_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if texture is None or normal is None or mask is None:
        missing = [p for p, img in [(texture_path, texture), (normal_path, normal), (mask_path, mask)] if img is None]
        raise FileNotFoundError(f"Missing files for {sample_id}: {missing}")

    # Convert BGR -> RGB for augmentation
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
    return texture, normal, mask


def save_triplet(out_id, texture_rgb, normal_rgb, mask):
    tex_name = f"{out_id}_CAM_Texture.jpg"
    norm_name = f"{out_id}_CAM_Normal.jpg"
    mask_name = f"{out_id}_CAM_Texture_mask.png"

    tex_path = os.path.join(OUT_IMG_DIR, tex_name)
    norm_path = os.path.join(OUT_IMG_DIR, norm_name)
    mask_path = os.path.join(OUT_MASK_DIR, mask_name)

    # Convert RGB -> BGR for saving
    cv2.imwrite(tex_path, cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(norm_path, cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, mask.astype(np.uint8))


def build_augmentation():
    # No cropping/resizing. Mild rotations, flips, and photometric changes.
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.3),
            albu.RandomRotate90(p=0.8),
            albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            albu.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
            albu.GaussNoise(var_limit=(10.0, 40.0), p=0.4),
            albu.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
        ],
        additional_targets={'normal': 'image', 'mask': 'mask'}
    )


def main():
    ensure_dirs()
    sample_ids = list_sample_ids(Config.TRAIN_IMG_DIR)
    original_ids = [sid for sid in sample_ids if '_aug' not in sid]

    if not original_ids:
        print("No original training images found; aborting.")
        return

    print(f"Original train images (filtered, no _aug): {len(original_ids)}")
    print(f"Augmentations per image: {NUM_AUG_PER_IMAGE}")
    print(f"Output folder: {OUT_TRAIN_DIR}")

    aug = build_augmentation()

    total_written = 0
    for sid in tqdm(original_ids, desc="Augmenting (no-crop, originals only)"):
        texture, normal, mask = load_triplet(sid)

        # Save original copy to the new folder
        save_triplet(sid, texture, normal, mask)
        total_written += 1

        # Generate augmented variants
        for j in range(NUM_AUG_PER_IMAGE):
            sample = aug(image=texture, normal=normal, mask=mask)
            tex_aug, norm_aug, mask_aug = sample['image'], sample['normal'], sample['mask']
            out_id = f"{sid}_aug{j+1:02d}"
            save_triplet(out_id, tex_aug, norm_aug, mask_aug)
            total_written += 1

    print(f"Done. Total images (with originals) written to new folder: {total_written}")
    print("Note: Validation set unchanged.")


if __name__ == "__main__":
    main()
