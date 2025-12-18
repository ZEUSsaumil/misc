import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import albumentations as albu

try:
    from config import Config
    from dataset import get_training_augmentation
except ImportError:
    from .config import Config
    from .dataset import get_training_augmentation

SRC_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'raw dataset', '20251201182754114464_-1')
SRC_IMG_DIR = os.path.join(SRC_BASE, 'rAW')
SRC_MASK_DIR = os.path.join(SRC_BASE, 'masks_multiclass')

DST_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw_aug95')
DST_IMG_DIR = os.path.join(DST_BASE, 'images')
DST_MASK_DIR = os.path.join(DST_BASE, 'masks_multiclass')

os.makedirs(DST_IMG_DIR, exist_ok=True)
os.makedirs(DST_MASK_DIR, exist_ok=True)

# Modalities we will handle
MODALITIES = ['texture', 'normal', 'shape1', 'shape2', 'shape3']


def list_sample_ids(images_dir):
    return sorted(
        [f.replace('_CAM_Texture.jpg', '') for f in os.listdir(images_dir) if f.endswith('_CAM_Texture.jpg')]
    )


def read_modalities(sample_id):
    imgs = {}
    paths = {
        'texture': os.path.join(SRC_IMG_DIR, f"{sample_id}_CAM_Texture.jpg"),
        'normal': os.path.join(SRC_IMG_DIR, f"{sample_id}_CAM_Normal.jpg"),
        'shape1': os.path.join(SRC_IMG_DIR, f"{sample_id}_CAM_Shape1.jpg"),
        'shape2': os.path.join(SRC_IMG_DIR, f"{sample_id}_CAM_Shape2.jpg"),
        'shape3': os.path.join(SRC_IMG_DIR, f"{sample_id}_CAM_Shape3.jpg"),
    }
    for k, p in paths.items():
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE if k.startswith('shape') else cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Missing modality {k} at {p}")
        if k == 'texture' or k == 'normal':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.expand_dims(img, axis=-1)
        imgs[k] = img
    return imgs


def read_mask(sample_id):
    p = os.path.join(SRC_MASK_DIR, f"{sample_id}_CAM_Texture_mask.png")
    mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Missing mask at {p}")
    return mask


def save_modalities(sample_id, suffix, imgs):
    for k, img in imgs.items():
        name = f"{sample_id}{suffix}_CAM_{'Texture' if k=='texture' else ('Normal' if k=='normal' else 'Shape'+k[-1])}.jpg"
        out_path = os.path.join(DST_IMG_DIR, name)
        if k == 'texture' or k == 'normal':
            cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(out_path, img.squeeze())


def save_mask(sample_id, suffix, mask):
    name = f"{sample_id}{suffix}_CAM_Texture_mask.png"
    out_path = os.path.join(DST_MASK_DIR, name)
    cv2.imwrite(out_path, mask.astype(np.uint8))


def main(target_total=95):
    sample_ids = list_sample_ids(SRC_IMG_DIR)
    orig_count = len(sample_ids)
    if orig_count == 0:
        print("No source images found.")
        return

    # Copy originals first
    for sid in sample_ids:
        imgs = read_modalities(sid)
        mask = read_mask(sid)
        save_modalities(sid, '', imgs)
        save_mask(sid, '', mask)

    needed = target_total - orig_count
    if needed <= 0:
        print(f"Already have {orig_count} samples; target {target_total} reached.")
        return

    base_aug = needed // orig_count
    remainder = needed % orig_count

    print(f"Original: {orig_count}, target: {target_total}, per-image aug: {base_aug} (+1 for first {remainder})")

    color_aug, geom_aug = get_training_augmentation(MODALITIES)

    total_generated = 0
    for idx, sid in enumerate(tqdm(sample_ids, desc="Augmenting")):
        num_aug = base_aug + (1 if idx < remainder else 0)
        imgs = read_modalities(sid)
        mask = read_mask(sid)

        for j in range(num_aug):
            # deep copies
            tex = imgs['texture'].copy()
            norm = imgs['normal'].copy()
            sh1 = imgs['shape1'].copy()
            sh2 = imgs['shape2'].copy()
            sh3 = imgs['shape3'].copy()
            msk = mask.copy()

            # color aug on texture only
            if color_aug:
                tex = color_aug(image=tex)['image']

            # geometric aug on all modalities + mask
            if geom_aug:
                sample = geom_aug(image=tex, normal=norm, shape1=sh1, shape2=sh2, shape3=sh3, mask=msk)
                tex = sample['image']
                norm = sample['normal']
                sh1 = sample['shape1']
                sh2 = sample['shape2']
                sh3 = sample['shape3']
                msk = sample['mask']

            out_imgs = {
                'texture': tex,
                'normal': norm,
                'shape1': sh1,
                'shape2': sh2,
                'shape3': sh3,
            }
            save_modalities(sid, f"_aug{j+1:02d}", out_imgs)
            save_mask(sid, f"_aug{j+1:02d}", msk)
            total_generated += 1

    final_total = orig_count + total_generated
    print(f"Generated {total_generated} augmented samples. Final count: {final_total}")


if __name__ == '__main__':
    main()
