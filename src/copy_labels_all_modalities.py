import os
import json
import glob
import re
from shutil import copyfile

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(BASE_DIR, 'raw dataset', '20251201182754114464_-1', 'LABELED TEXTURE FILES')
RAW_DIR = os.path.join(BASE_DIR, 'raw dataset', '20251201182754114464_-1', 'rAW')
OUT_DIR = os.path.join(BASE_DIR, 'raw dataset', '20251201182754114464_-1', 'LABELED_ALL_MODALITIES')
SOURCE_MODALITY = 'Texture'  # we have Texture JSONs labeled
ANGLE_PREFIX_PATTERN = r'(.*)_CAM_'

TARGET_MODALITIES = ['Normal', 'Shape1', 'Shape2', 'Shape3']


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    json_paths = glob.glob(os.path.join(JSON_DIR, '*.json'))
    print(f"Found {len(json_paths)} source JSONs")

    # Group by angle prefix (sample id without modality suffix)
    groups = {}
    for jp in json_paths:
        fn = os.path.basename(jp)
        m = re.match(ANGLE_PREFIX_PATTERN, fn)
        if not m:
            print(f"Skip (no match): {fn}")
            continue
        prefix = m.group(1)  # e.g., 20251201_..._T0000_01
        groups[prefix] = jp

    total_written = 0
    for prefix, src_path in groups.items():
        with open(src_path, 'r', encoding='utf-8') as f:
            src_data = json.load(f)
        shapes = src_data.get('shapes', [])
        h = src_data.get('imageHeight')
        w = src_data.get('imageWidth')

        # Write the source copy to OUT_DIR
        src_out = os.path.join(OUT_DIR, os.path.basename(src_path))
        copyfile(src_path, src_out)
        total_written += 1

        # For each target modality, build filename and copy shapes
        for mod in TARGET_MODALITIES:
            target_name = f"{prefix}_CAM_{mod}.json"
            target_out = os.path.join(OUT_DIR, target_name)

            target_src = os.path.join(RAW_DIR, target_name)
            base_data = None
            if os.path.exists(target_src):
                try:
                    with open(target_src, 'r', encoding='utf-8') as f:
                        base_data = json.load(f)
                except json.JSONDecodeError:
                    base_data = None

            if base_data is None:
                # create minimal labelme-like structure
                base_data = {
                    "version": "5.5.0",
                    "flags": {},
                    "shapes": [],
                    "imagePath": f"..\\{prefix}_CAM_{mod}.jpg",
                    "imageData": None,
                    "imageHeight": h,
                    "imageWidth": w,
                }

            base_data['shapes'] = shapes
            base_data['imageHeight'] = h
            base_data['imageWidth'] = w
            base_data['imagePath'] = f"..\\{prefix}_CAM_{mod}.jpg"

            with open(target_out, 'w', encoding='utf-8') as f:
                json.dump(base_data, f, indent=4)
            total_written += 1

    print(f"Done. Wrote {total_written} labeled JSONs to {OUT_DIR}")


if __name__ == '__main__':
    main()
