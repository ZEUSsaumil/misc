"""
Self-supervised SimCLR-style pretraining for the EfficientNet encoder on unlabeled
industrial images (normal/texture/shape). This script prepares the components so you
can launch pretraining when ready; it does not auto-run any training loop unless you
call main().

Usage (when you choose to run):
    python -m src.selfsup_pretrain --modalities normal texture \
        --train-img-dir data/hybrid/train/images --val-img-dir data/hybrid/val/images \
        --epochs 50 --batch-size 32 --encoder efficientnet-b4

The script:
    - Builds an unlabeled dataset over the provided image dirs.
    - Generates two strong augmentations per sample (SimCLR view pair).
    - Trains a projection head on top of the encoder with NT-Xent loss.
    - Saves encoder weights to models/selfsup_encoder_<tag>.pth for downstream init.
"""

import os
import argparse
from pathlib import Path
from typing import List, Sequence

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

try:
    from config import Config
except ImportError:
    from .config import Config


def _read_modality(images_dir: Path, sample_id: str, modality: str) -> np.ndarray:
    """Load one modality; keeps channel dimension."""
    if modality == 'texture':
        path = images_dir / f"{sample_id}_CAM_Texture.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if modality == 'normal':
        path = images_dir / f"{sample_id}_CAM_Normal.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if modality in ('shape1', 'shape2', 'shape3'):
        suffix = modality[-1]
        path = images_dir / f"{sample_id}_CAM_Shape{suffix}.jpg"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return np.expand_dims(img, axis=-1)
    raise ValueError(f"Unknown modality {modality}")


class UnlabeledMultiModalDataset(Dataset):
    """Returns two augmented views of stacked modalities for SimCLR."""

    def __init__(self, images_dirs: Sequence[Path], modalities: List[str], transform):
        self.images_dirs = [Path(p) for p in images_dirs if Path(p).exists()]
        self.modalities = modalities
        self.transform = transform

        ids = []
        for img_dir in self.images_dirs:
            for fname in os.listdir(img_dir):
                if fname.endswith('_CAM_Texture.jpg'):
                    ids.append(fname.replace('_CAM_Texture.jpg', ''))
        self.ids = sorted(set(ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        # load modalities and stack
        mod_arrays = []
        for m in self.modalities:
            mod = _read_modality(self._pick_dir(sample_id), sample_id, m)
            mod_arrays.append(mod)
        image = np.concatenate(mod_arrays, axis=-1).astype('float32')

        # create two augmented views
        view1 = self._apply_aug(image)
        view2 = self._apply_aug(image)
        return view1, view2

    def _pick_dir(self, sample_id: str) -> Path:
        # choose the first directory containing the sample
        for d in self.images_dirs:
            if (d / f"{sample_id}_CAM_Texture.jpg").exists():
                return d
        return self.images_dirs[0]

    def _apply_aug(self, image: np.ndarray) -> torch.Tensor:
        out = self.transform(image=image)['image']
        return torch.from_numpy(out)


def build_simclr_transform(modalities: List[str]):
    ch_total = sum(Config.MODALITY_CHANNELS[m] for m in modalities)
    aug = albu.Compose([
        albu.RandomResizedCrop(height=320, width=320, scale=(0.6, 1.0), ratio=(0.8, 1.25), p=1.0),
        albu.HorizontalFlip(p=0.5),
        albu.OneOf([
            albu.GaussianBlur(blur_limit=5, p=1),
            albu.MotionBlur(blur_limit=5, p=1),
        ], p=0.5),
        albu.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        albu.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        albu.Normalize(mean=[0.5]*ch_total, std=[0.5]*ch_total, max_pixel_value=255.0),
        albu.Lambda(image=lambda x, **k: x.transpose(2, 0, 1)),
    ])
    return aug


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, encoder_name: str, in_channels: int, proj_dim: int = 256):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=in_channels, depth=5, weights=None)
        feat_dim = self.encoder.out_channels[-1]
        self.proj = ProjectionHead(feat_dim, proj_dim)

    def forward(self, x):
        feats = self.encoder(x)[-1]  # B x C x H x W
        pooled = torch.mean(feats, dim=[2, 3])
        z = self.proj(pooled)
        return nn.functional.normalize(z, dim=1)


def ntxent_loss(z1, z2, temperature=0.1):
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    bs = z1.size(0)
    labels = torch.cat([torch.arange(bs), torch.arange(bs)], dim=0).to(z.device)

    # mask self-similarity
    mask = torch.eye(2 * bs, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)

    logits = sim
    labels = labels
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss


def train_ssl(args):
    modalities = args.modalities
    in_channels = sum(Config.MODALITY_CHANNELS[m] for m in modalities)
    transform = build_simclr_transform(modalities)

    dataset = UnlabeledMultiModalDataset(
        images_dirs=[Path(args.train_img_dir), Path(args.val_img_dir)],
        modalities=modalities,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimCLR(args.encoder, in_channels, proj_dim=args.proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    Config.ensure_dirs()
    tag = args.run_tag or f"ssl-{'-'.join(modalities)}-{args.encoder}"
    save_path = os.path.join(Config.MODEL_SAVE_DIR, f"selfsup_encoder_{tag}.pth")

    print(f"[SSL] Samples: {len(dataset)}, save: {save_path}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for v1, v2 in loader:
            v1 = v1.float().to(device)
            v2 = v2.float().to(device)
            z1 = model(v1)
            z2 = model(v2)
            loss = ntxent_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch+1}/{args.epochs} | loss {avg_loss:.4f}")

    torch.save(model.encoder.state_dict(), save_path)
    print(f"Saved encoder weights to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised SimCLR pretrain")
    parser.add_argument('--train-img-dir', default=Config.TRAIN_IMG_DIR)
    parser.add_argument('--val-img-dir', default=Config.VAL_IMG_DIR)
    parser.add_argument('--modalities', nargs='+', default=['normal', 'texture'], help='Modalities to stack (order matters)')
    parser.add_argument('--encoder', default=Config.ENCODER)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--proj-dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--run-tag', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    train_ssl(args)


if __name__ == '__main__':
    main()