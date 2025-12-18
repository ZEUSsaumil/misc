import os
import cv2
import numpy as np
from tqdm import tqdm

COLOR_MAP = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (0, 122, 0),
    3: (160, 170, 250),
    4: (0, 255, 0),
}

def colorize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        m = img == cls
        if m.any():
            rgb[m] = color
    return rgb


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(base, 'raw dataset', '20251201182754114464_-1', 'masks_multiclass_all')
    dst = os.path.join(base, 'raw dataset', '20251201182754114464_-1', 'masks_multiclass_all_color')
    os.makedirs(dst, exist_ok=True)

    files = [f for f in os.listdir(src) if f.lower().endswith('.png')]
    print(f'Found {len(files)} masks')

    for f in tqdm(files, desc='Colorizing'):
        p = os.path.join(src, f)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'Warning: cannot read {p}')
            continue
        rgb = colorize(img)
        cv2.imwrite(os.path.join(dst, f), rgb)


if __name__ == '__main__':
    main()
