import os
import math
import cv2
import numpy as np
from tqdm import tqdm

# Use the same augmentation definitions as training
from config import Config
from dataset import get_training_augmentation


def list_sample_ids(images_dir):
    return sorted(
        [f.replace('_CAM_Texture.jpg', '') for f in os.listdir(images_dir) if f.endswith('_CAM_Texture.jpg')]
    )


def ensure_dirs():
    os.makedirs(Config.TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(Config.TRAIN_MASK_DIR, exist_ok=True)


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


def save_triplet(sample_id, aug_idx, texture_rgb, normal_rgb, mask):
    tex_name = f"{sample_id}_aug{aug_idx:02d}_CAM_Texture.jpg"
    norm_name = f"{sample_id}_aug{aug_idx:02d}_CAM_Normal.jpg"
    mask_name = f"{sample_id}_aug{aug_idx:02d}_CAM_Texture_mask.png"

    tex_path = os.path.join(Config.TRAIN_IMG_DIR, tex_name)
    norm_path = os.path.join(Config.TRAIN_IMG_DIR, norm_name)
    mask_path = os.path.join(Config.TRAIN_MASK_DIR, mask_name)

    # Convert RGB -> BGR for saving
    cv2.imwrite(tex_path, cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(norm_path, cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, mask.astype(np.uint8))


def main(target_total=500):
    ensure_dirs()
    sample_ids = list_sample_ids(Config.TRAIN_IMG_DIR)
    orig_count = len(sample_ids)
    if orig_count == 0:
        print("No training images found. Aborting.")
        return

    needed = target_total - orig_count
    if needed <= 0:
        print(f"Already have {orig_count} images, target {target_total}; no augmentation needed.")
        return

    base_aug = needed // orig_count
    remainder = needed % orig_count

    print(f"Original train images: {orig_count}")
    print(f"Target total images: {target_total}")
    print(f"Augmentations per image: {base_aug} (+1 for first {remainder} samples)")

    color_aug, geom_aug = get_training_augmentation()

    total_generated = 0
    for idx, sid in enumerate(tqdm(sample_ids, desc="Augmenting")):
        num_aug = base_aug + (1 if idx < remainder else 0)
        texture, normal, mask = load_triplet(sid)

        for j in range(num_aug):
            tex_aug = texture.copy()
            norm_aug = normal.copy()
            mask_aug = mask.copy()

            # Apply color aug to texture only
            if color_aug:
                tex_aug = color_aug(image=tex_aug)['image']

            # Apply geometric aug to all three
            if geom_aug:
                sample = geom_aug(image=tex_aug, normal=norm_aug, mask=mask_aug)
                tex_aug, norm_aug, mask_aug = sample['image'], sample['normal'], sample['mask']

            save_triplet(sid, j + 1, tex_aug, norm_aug, mask_aug)
            total_generated += 1

    final_total = orig_count + total_generated
    print(f"Generated {total_generated} augmented samples. Final train set size: {final_total}")


if __name__ == "__main__":
    main()
