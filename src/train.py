import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np

# Fix imports for script execution
try:
    from config import Config
    from dataset import SomicDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing
    from model import get_model
except ImportError:
    from .config import Config
    from .dataset import SomicDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing
    from .model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Somic segmentation model")
    parser.add_argument('--train-img-dir', default=Config.TRAIN_IMG_DIR)
    parser.add_argument('--train-mask-dir', default=Config.TRAIN_MASK_DIR)
    parser.add_argument('--val-img-dir', default=Config.VAL_IMG_DIR)
    parser.add_argument('--val-mask-dir', default=Config.VAL_MASK_DIR)
    parser.add_argument('--modalities', nargs='+', default=Config.INPUT_MODALITIES, help='Modalities order, e.g., texture normal')
    parser.add_argument('--use-view-id', action='store_true', default=Config.USE_VIEW_ID, help='Append normalized view-id channel')
    parser.add_argument('--classes', nargs='+', default=Config.CLASSES, help='Class list')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--num-workers', type=int, default=Config.NUM_WORKERS)
    parser.add_argument('--encoder', default=Config.ENCODER)
    parser.add_argument('--encoder-weights', default=Config.ENCODER_WEIGHTS)
    parser.add_argument('--run-tag', default=None, help='Optional suffix for saved model files')
    parser.add_argument('--encoder-checkpoint', default=None, help='Path to encoder state_dict from self-supervised pretrain')
    return parser.parse_args()

def train():
    args = parse_args()
    Config.ensure_dirs()

    # Apply runtime overrides
    Config.TRAIN_IMG_DIR = args.train_img_dir
    Config.TRAIN_MASK_DIR = args.train_mask_dir
    Config.VAL_IMG_DIR = args.val_img_dir
    Config.VAL_MASK_DIR = args.val_mask_dir
    Config.INPUT_MODALITIES = args.modalities
    Config.USE_VIEW_ID = args.use_view_id
    Config.CLASSES = args.classes
    Config.NUM_EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    Config.NUM_WORKERS = args.num_workers
    Config.ENCODER = args.encoder
    Config.ENCODER_WEIGHTS = args.encoder_weights

    modalities_for_stack = list(Config.INPUT_MODALITIES)
    if Config.USE_VIEW_ID and 'view_id' not in modalities_for_stack:
        modalities_for_stack = modalities_for_stack + ['view_id']

    Config.IN_CHANNELS = Config.compute_in_channels(modalities_for_stack, use_view_id=Config.USE_VIEW_ID)

    run_tag = args.run_tag or f"{'-'.join(Config.INPUT_MODALITIES)}{'-view' if Config.USE_VIEW_ID else ''}"
    save_best_path = os.path.join(Config.MODEL_SAVE_DIR, f"best_model_{run_tag}.pth")
    save_latest_path = os.path.join(Config.MODEL_SAVE_DIR, f"latest_model_{run_tag}.pth")

    # 1. Dataset & Dataloader
    preprocessing_fn = smp.encoders.get_preprocessing_fn(Config.ENCODER, Config.ENCODER_WEIGHTS)
    
    train_dataset = SomicDataset(
        Config.TRAIN_IMG_DIR, 
        Config.TRAIN_MASK_DIR, 
        augmentation=get_training_augmentation(modalities_for_stack), 
        preprocessing=get_preprocessing(preprocessing_fn, modalities_for_stack),
        classes=Config.CLASSES,
        modalities=modalities_for_stack,
        use_view_id=Config.USE_VIEW_ID,
    )
    
    valid_dataset = SomicDataset(
        Config.VAL_IMG_DIR, 
        Config.VAL_MASK_DIR, 
        augmentation=get_validation_augmentation(modalities_for_stack), 
        preprocessing=get_preprocessing(preprocessing_fn, modalities_for_stack),
        classes=Config.CLASSES,
        modalities=modalities_for_stack,
        use_view_id=Config.USE_VIEW_ID,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # 2. Model
    model = get_model()
    model.to(Config.DEVICE)

    # Optional: load self-supervised encoder checkpoint
    if args.encoder_checkpoint:
        state = torch.load(args.encoder_checkpoint, map_location=Config.DEVICE)
        missing, unexpected = model.encoder.load_state_dict(state, strict=False)
        print(f"Loaded encoder checkpoint: {args.encoder_checkpoint}")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    
    # 3. Loss, Optimizer
    loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 4. Training Loop
    max_score = 0
    
    print(f"Starting training for {Config.NUM_EPOCHS} epochs on {Config.DEVICE}...")
    print(f"Modalities: {Config.INPUT_MODALITIES}, view-id: {Config.USE_VIEW_ID}, classes: {Config.CLASSES}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(valid_dataset)}")
    
    for epoch in range(Config.NUM_EPOCHS):
        
        # Training
        model.train()
        train_loss = 0
        train_iou = 0
        
        # pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]")
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]")
        for batch_idx, (images, masks) in enumerate(train_loader):
            print(f"Batch {batch_idx} started")
            try:
                images = images.to(Config.DEVICE)
                masks = masks.to(Config.DEVICE).long().squeeze(1) # Masks should be (B, H, W) for multiclass loss
                
                optimizer.zero_grad()
                
                # Forward
                # print("Forward pass...")
                logits = model(images)
                
                # Loss
                # print("Loss calculation...")
                loss = loss_fn(logits, masks)
                
                # Backward
                # print("Backward pass...")
                loss.backward()
                optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                
                # Simple IoU calculation
                # Logits: (B, C, H, W)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                tp, fp, fn, tn = smp.metrics.get_stats(preds, masks, mode='multiclass', num_classes=len(Config.CLASSES))
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                train_iou += iou.item()
                
                print(f"Batch {batch_idx} done. Loss: {loss.item():.4f}, IoU: {iou.item():.4f}")
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
            
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad():
            pbar_val = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]")
            for images, masks in pbar_val:
                images = images.to(Config.DEVICE)
                masks = masks.to(Config.DEVICE).long().squeeze(1)
                
                logits = model(images)
                loss = loss_fn(logits, masks)
                
                val_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                tp, fp, fn, tn = smp.metrics.get_stats(preds, masks, mode='multiclass', num_classes=len(Config.CLASSES))
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                val_iou += iou.item()
                
                pbar_val.set_postfix({'loss': loss.item(), 'iou': iou.item()})
                
        val_loss /= len(valid_loader)
        val_iou /= len(valid_loader)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > max_score:
            max_score = val_iou
            torch.save(model.state_dict(), save_best_path)
            print(f'New best model saved at {save_best_path}!')
            
        # Save latest model
        torch.save(model.state_dict(), save_latest_path)
        
        # Learning rate decay
        if epoch == 25:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
            print('Decreased learning rate to 1e-5!')

if __name__ == '__main__':
    train()
