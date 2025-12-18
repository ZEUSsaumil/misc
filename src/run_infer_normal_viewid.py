import os
import sys
import torch
import numpy as np
import cv2

sys.path.append(os.path.dirname(__file__))

import segmentation_models_pytorch as smp
from config import Config
from dataset import SomicDataset, get_preprocessing, get_validation_augmentation
from model import get_model


def main():
    Config.INPUT_MODALITIES = ['normal', 'view_id']
    Config.USE_VIEW_ID = True
    modalities = Config.INPUT_MODALITIES
    Config.IN_CHANNELS = Config.compute_in_channels(modalities, use_view_id=True)

    img_dir = os.path.join('data', 'hybrid', 'val', 'images')
    mask_dir = os.path.join('data', 'hybrid', 'val', 'masks_multiclass')
    model_path = os.path.join('models', 'best_model_normal-view-full.pth')
    out_dir = 'inference_results'
    os.makedirs(out_dir, exist_ok=True)

    model = get_model()
    state = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.to(Config.DEVICE)
    model.eval()

    preproc_fn = smp.encoders.get_preprocessing_fn(Config.ENCODER, Config.ENCODER_WEIGHTS)
    val_aug = get_validation_augmentation(modalities)
    val_ds = SomicDataset(
        img_dir,
        mask_dir,
        augmentation=val_aug,
        preprocessing=get_preprocessing(preproc_fn, modalities),
        classes=Config.CLASSES,
        modalities=modalities,
        use_view_id=True,
    )

    idx = 0
    image_np, mask_np = val_ds[idx]
    with torch.no_grad():
        x = torch.from_numpy(image_np).unsqueeze(0).to(Config.DEVICE)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    tex = image_np[:3].transpose(1, 2, 0)
    tex_min, tex_max = tex.min(), tex.max()
    tex = (tex - tex_min) / (tex_max - tex_min + 1e-8)
    tex_uint8 = (tex * 255).astype(np.uint8)
    color = np.zeros((*pred.shape, 3), dtype=np.uint8)
    color[pred == 1] = [128, 128, 128]
    color[pred == 2] = [255, 0, 0]
    color[pred == 3] = [0, 255, 0]
    color[pred == 4] = [0, 0, 255]
    overlay = cv2.addWeighted(tex_uint8, 0.7, color, 0.3, 0)

    out_png = os.path.join(out_dir, 'normal_viewid_val0_overlay.png')
    cv2.imwrite(out_png, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(out_dir, 'normal_viewid_val0_pred.npy'), pred)
    np.save(os.path.join(out_dir, 'normal_viewid_val0_gt.npy'), mask_np.squeeze())

    print('Saved:', os.path.abspath(out_png))
    print('GT class counts:', {int(k): int(v) for k, v in zip(*np.unique(mask_np, return_counts=True))})
    print('Pred class counts:', {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))})
    print('Shapes:', image_np.shape, mask_np.shape, pred.shape)


if __name__ == '__main__':
    main()
