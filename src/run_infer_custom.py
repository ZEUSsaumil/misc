import os
import sys
import cv2
import torch
import numpy as np

sys.path.append(os.path.dirname(__file__))

import segmentation_models_pytorch as smp
from config import Config
from model import get_model


def run(img_path, out_name='input_overlay.png', view_id_val=0.0):
    model_path = os.path.join('models', 'best_model_normal-view-full.pth')
    out_dir = 'inference_results'
    os.makedirs(out_dir, exist_ok=True)

    Config.INPUT_MODALITIES = ['normal', 'view_id']
    Config.USE_VIEW_ID = True
    modalities = Config.INPUT_MODALITIES
    Config.IN_CHANNELS = Config.compute_in_channels(modalities, use_view_id=True)

    bgr = cv2.imread(img_path)
    if bgr is None:
        raise SystemExit(f'Image not found: {img_path}')
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pad_h = (32 - orig_h % 32) % 32
    pad_w = (32 - orig_w % 32) % 32
    if pad_h or pad_w:
        rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    h, w = rgb.shape[:2]
    view_id = np.full((h, w, 1), view_id_val, dtype=np.float32)
    stack = np.concatenate([rgb.astype(np.float32), view_id], axis=-1)

    preproc_fn = smp.encoders.get_preprocessing_fn(Config.ENCODER, Config.ENCODER_WEIGHTS)
    rgb_proc = preproc_fn(rgb.astype(np.float32))
    stack[:, :, :3] = rgb_proc
    stack[:, :, 3] = view_id[:, :, 0]
    chw = stack.transpose(2, 0, 1)

    model = get_model()
    state = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.to(Config.DEVICE)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(chw).unsqueeze(0).to(Config.DEVICE)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred = pred[:orig_h, :orig_w]
    tex = rgb[:orig_h, :orig_w].astype(np.float32)
    tex_min, tex_max = tex.min(), tex.max()
    tex = (tex - tex_min) / (tex_max - tex_min + 1e-8)
    tex_uint8 = (tex * 255).astype(np.uint8)
    color = np.zeros((*pred.shape, 3), dtype=np.uint8)
    color[pred == 1] = [128, 128, 128]
    color[pred == 2] = [255, 0, 0]
    color[pred == 3] = [0, 255, 0]
    color[pred == 4] = [0, 0, 255]
    overlay = cv2.addWeighted(tex_uint8, 0.7, color, 0.3, 0)

    out_png = os.path.join(out_dir, out_name)
    cv2.imwrite(out_png, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(out_dir, out_name.replace('.png', '_pred.npy')), pred)

    print('Saved overlay to', os.path.abspath(out_png))
    print('Pred class counts:', {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))})


def main():
    img_path = os.path.join('inference_custom', 'input.jpg')
    run(img_path)


if __name__ == '__main__':
    main()
