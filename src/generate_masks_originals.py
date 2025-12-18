import os
import json
import glob
import cv2
import numpy as np

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


def generate_masks_originals():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'raw dataset', '20251201182754114464_-1', 'rAW')
    json_dir = os.path.join(base_dir, 'raw dataset', '20251201182754114464_-1', 'LABELED TEXTURE FILES')
    out_dir = os.path.join(base_dir, 'raw dataset', '20251201182754114464_-1', 'masks_multiclass')
    os.makedirs(out_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    print(f"Found {len(json_files)} JSON files.")

    for jp in json_files:
        base = os.path.basename(jp).replace('.json', '')
        img_path = os.path.join(raw_dir, base + '.jpg')
        if not os.path.exists(img_path):
            print(f"Warning: image not found for {base}")
            continue

        with open(jp, 'r') as f:
            data = json.load(f)

        h = data.get('imageHeight')
        w = data.get('imageWidth')
        if h is None or w is None:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: cannot read image {img_path}")
                continue
            h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        shapes = data.get('shapes', [])

        # part first
        for shape in shapes:
            lbl = shape['label']
            pts = np.array(shape['points'], dtype=np.int32)
            cls_id = _label_to_class_id(lbl)
            if cls_id == 1:
                cv2.fillPoly(mask, [pts], cls_id)

        # defects overwrite
        for shape in shapes:
            lbl = shape['label']
            pts = np.array(shape['points'], dtype=np.int32)
            cls_id = _label_to_class_id(lbl)
            if cls_id != 1:
                cv2.fillPoly(mask, [pts], cls_id)

        out_path = os.path.join(out_dir, base + '_mask.png')
        cv2.imwrite(out_path, mask)
        uniq = np.unique(mask)
        if uniq.size <= 1:
            print(f"Warning: {os.path.basename(out_path)} has values {uniq}")
        else:
            print(f"Saved {os.path.basename(out_path)} values {uniq}")


if __name__ == '__main__':
    generate_masks_originals()
