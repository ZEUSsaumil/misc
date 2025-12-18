import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu

try:
    from config import Config
except ImportError:
    from .config import Config

class SomicDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None, classes=None, modalities=None, use_view_id=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.modalities = modalities or Config.INPUT_MODALITIES
        self.use_view_id = use_view_id or ('view_id' in self.modalities)

        # Identify samples by texture filenames (canonical key)
        all_files = os.listdir(images_dir)
        self.ids = [f.replace('_CAM_Texture.jpg', '') for f in all_files if '_CAM_Texture.jpg' in f]

        # convert str names to class values on masks
        if classes:
            try:
                self.class_values = [classes.index(cls.lower()) for cls in classes]
            except ValueError:
                self.class_values = list(range(len(classes)))
        else:
            self.class_values = [0]
        
        # Map view-id tokens to normalized float channel if requested
        self.view_id_value = {}
        if self.use_view_id:
            tokens = []
            for sid in self.ids:
                parts = sid.split('_')
                tokens.append(parts[-1] if parts else '0')
            uniq = sorted(set(tokens))
            token_to_val = {tok: idx / max(1, (len(uniq) - 1)) if len(uniq) > 1 else 0.5 for idx, tok in enumerate(uniq)}
            for sid in self.ids:
                tok = sid.split('_')[-1] if '_' in sid else sid
                self.view_id_value[sid] = token_to_val.get(tok, 0.0)
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def _read_modality(self, sample_id, modality):
        """Load a modality and return np.array with channel dimension preserved."""
        if modality == 'texture':
            path = os.path.join(self.images_dir, f"{sample_id}_CAM_Texture.jpg")
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Texture not found: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if modality == 'normal':
            path = os.path.join(self.images_dir, f"{sample_id}_CAM_Normal.jpg")
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Normal not found: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if modality in ('shape1', 'shape2', 'shape3'):
            suffix = modality[-1]  # '1','2','3'
            path = os.path.join(self.images_dir, f"{sample_id}_CAM_Shape{suffix}.jpg")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Shape{suffix} not found: {path}")
            return np.expand_dims(img, axis=-1)
        raise ValueError(f"Unknown modality: {modality}")

    def __getitem__(self, i):
        sample_id = self.ids[i]

        # Load mask (canonical texture mask for alignment)
        mask_path = os.path.join(self.masks_dir, f"{sample_id}_CAM_Texture_mask.png")
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            # Fallback: create empty mask
            # Use first modality height/width to size mask
            first_mod = self._read_modality(sample_id, self.modalities[0])
            mask = np.zeros(first_mod.shape[:2], dtype=np.uint8)

        # Multiclass mask already encoded: 0 bg, 1 part, 2 scratch, 3 dent, 4 chip
        mask = mask.astype('float')
        mask = np.expand_dims(mask, axis=-1)

        # Load modalities
        mod_arrays = {}
        for m in self.modalities:
            if m == 'view_id':
                continue
            mod_arrays[m] = self._read_modality(sample_id, m)

        # Synthesized view-id channel if requested
        if self.use_view_id:
            base = next(iter(mod_arrays.values()))
            h, w = base.shape[:2]
            v = float(self.view_id_value.get(sample_id, 0.0))
            mod_arrays['view_id'] = np.full((h, w, 1), v, dtype=np.float32)

        # Apply augmentations
        if self.augmentation:
            if isinstance(self.augmentation, (list, tuple)) and len(self.augmentation) == 2:
                color_aug, geometric_aug = self.augmentation

                # Color on texture only
                if color_aug and 'texture' in mod_arrays:
                    sample = color_aug(image=mod_arrays['texture'])
                    mod_arrays['texture'] = sample['image']

                if geometric_aug:
                    base_img_key = 'texture' if 'texture' in mod_arrays else next(iter(mod_arrays.keys()))
                    aug_kwargs = {'image': mod_arrays.get(base_img_key), 'mask': mask}
                    for key in ['normal', 'shape1', 'shape2', 'shape3', 'view_id']:
                        if key in mod_arrays:
                            aug_kwargs[key] = mod_arrays[key]
                    sample = geometric_aug(**aug_kwargs)
                    # Retrieve back
                    if 'image' in sample and base_img_key in mod_arrays:
                        mod_arrays[base_img_key] = sample['image']
                    if 'normal' in sample:
                        mod_arrays['normal'] = sample['normal']
                    for key in ['shape1', 'shape2', 'shape3', 'view_id']:
                        if key in sample:
                            mod_arrays[key] = sample[key]
                    mask = sample['mask']
            else:
                # Single compose path
                base_img_key = 'texture' if 'texture' in mod_arrays else next(iter(mod_arrays.keys()))
                aug_kwargs = {'image': mod_arrays.get(base_img_key), 'mask': mask}
                for key in ['normal', 'shape1', 'shape2', 'shape3', 'view_id']:
                    if key in mod_arrays:
                        aug_kwargs[key] = mod_arrays[key]
                sample = self.augmentation(**aug_kwargs)
                if 'image' in sample and base_img_key in mod_arrays:
                    mod_arrays[base_img_key] = sample['image']
                if 'normal' in sample:
                    mod_arrays['normal'] = sample['normal']
                for key in ['shape1', 'shape2', 'shape3', 'view_id']:
                    if key in sample:
                        mod_arrays[key] = sample[key]
                mask = sample['mask']

        # Stack modalities in configured order
        stacked = []
        for m in self.modalities:
            stacked.append(mod_arrays[m])
        image = np.concatenate(stacked, axis=-1)

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

def _build_additional_targets(modalities):
    targets = {}
    for key in ['normal', 'shape1', 'shape2', 'shape3']:
        if key in modalities:
            targets[key] = 'image'
    if 'view_id' in modalities:
        targets['view_id'] = 'mask'
    return targets


def get_training_augmentation(modalities=None):
    modalities = modalities or Config.INPUT_MODALITIES

    # Color transforms (Texture only)
    color_transform = [
        albu.GaussNoise(p=0.2),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    
    # Geometric transforms (apply to all modalities + mask)
    geometric_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.Perspective(p=0.5),
    ]
    
    add_targets = _build_additional_targets(modalities)
    return (albu.Compose(color_transform), albu.Compose(geometric_transform, additional_targets=add_targets))

def get_validation_augmentation(modalities=None):
    modalities = modalities or Config.INPUT_MODALITIES
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32)
    ]
    add_targets = _build_additional_targets(modalities)
    return (None, albu.Compose(test_transform, additional_targets=add_targets))

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn, modalities=None):
    modalities = modalities or Config.INPUT_MODALITIES

    def preprocess_multi(image, **kwargs):
        # image HxWxC stacked in modality order
        image = image.astype('float32')
        out_slices = []
        offset = 0
        for m in modalities:
            ch = Config.MODALITY_CHANNELS.get(m, 1)
            sl = image[:, :, offset:offset+ch]
            offset += ch
            if m == 'view_id':
                # view-id channel already normalized 0..1
                out_slices.append(sl.astype('float32'))
                continue
            if ch == 3 and preprocessing_fn is not None:
                sl = preprocessing_fn(sl)
            else:
                sl = sl / 255.0
            out_slices.append(sl)
        return np.concatenate(out_slices, axis=-1)

    _transform = [
        albu.Lambda(image=preprocess_multi),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
