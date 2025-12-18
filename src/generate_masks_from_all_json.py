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


def label_to_class(label: str) -> int:
    lbl = label.strip().lower()
    if lbl in PART_LABELS:
        return 1
    if lbl in DEFECT_LABEL_MAP:
        return DEFECT_LABEL_MAP[lbl]
    for k, cid in DEFECT_LABEL_MAP.items():
        if k in lbl:
            return cid
    return 2


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_dir = os.path.join(base, 'raw dataset', '20251201182754114464_-1', 'LABELED_ALL_MODALITIES')
    out_gray = os.path.join(base, 'raw dataset', '20251201182754114464_-1', 'masks_multiclass_all')
    os.makedirs(out_gray, exist_ok=True)

    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    print(f'Found {len(json_files)} jsons')

    for jp in json_files:
        with open(jp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        h = data.get('imageHeight')
        w = data.get('imageWidth')
        if not h or not w:
            print(f'Warning: missing size in {jp}, skipped')
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        shapes = data.get('shapes', [])
        # part first
        for s in shapes:
            cls = label_to_class(s['label'])
            if cls == 1:
                pts = np.array(s['points'], dtype=np.int32)
                cv2.fillPoly(mask, [pts], cls)
        # defects overwrite
        for s in shapes:
            cls = label_to_class(s['label'])
            if cls != 1:
                pts = np.array(s['points'], dtype=np.int32)
                cv2.fillPoly(mask, [pts], cls)
        out_name = os.path.basename(jp).replace('.json', '_mask.png')
        cv2.imwrite(os.path.join(out_gray, out_name), mask)
    print('Done writing masks to', out_gray)


if __name__ == '__main__':
    main()
