import os
import cv2
import numpy as np

# Color map inspired by provided snippet; only classes 0-4 are used here.
COLOR_MAP = {
    0: (0, 0, 0),        # background
    1: (128, 64, 128),   # part / good
    2: (0, 122, 0),      # scratch
    3: (160, 170, 250),  # dent
    4: (0, 255, 0),      # chip
}


def colorize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        mask = img == cls
        if mask.any():
            rgb[mask] = color
    return rgb


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, 'raw dataset', '20251201182754114464_-1', 'masks_multiclass')
    out_dir = os.path.join(base_dir, 'raw dataset', '20251201182754114464_-1', 'masks_multiclass_color')
    os.makedirs(out_dir, exist_ok=True)

    mask_files = [f for f in os.listdir(src_dir) if f.lower().endswith('.png')]
    print(f'Found {len(mask_files)} masks')

    for fname in mask_files:
        src_path = os.path.join(src_dir, fname)
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'Warning: cannot read {src_path}')
            continue
        rgb = colorize(img)
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, rgb)
        uniq = np.unique(img)
        print(f'Saved {fname} (classes {uniq})')


if __name__ == '__main__':
    main()
