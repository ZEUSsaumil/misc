import os
import torch

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    # Hybrid split created from the 20251201182754114464_-1 raw set (copy-only, no aug)
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'hybrid', 'train', 'images')
    TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'hybrid', 'train', 'masks_multiclass')
    VAL_IMG_DIR = os.path.join(DATA_DIR, 'hybrid', 'val', 'images')
    VAL_MASK_DIR = os.path.join(DATA_DIR, 'hybrid', 'val', 'masks_multiclass')
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')

    # Model Parameters
    ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    # Modalities: texture, normal, shape1/2/3 (shape are single-channel); view_id is synthetic 1-channel
    INPUT_MODALITIES = ['texture', 'normal']
    MODALITY_CHANNELS = {'texture': 3, 'normal': 3, 'shape1': 1, 'shape2': 1, 'shape3': 1, 'view_id': 1}
    IN_CHANNELS = None  # computed at runtime based on selected modalities + view-id flag
    CLASSES = ['background', 'part', 'scratch', 'dent', 'chip']
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_VIEW_ID = False

    # Training Hyperparameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 2
    NUM_EPOCHS = 50
    IMAGE_SIZE = (512, 512) # Resize input images to this size
    NUM_WORKERS = 0  # Windows-friendly default; can be overridden

    # Thresholds
    CONFIDENCE_THRESHOLD = 0.5

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)

    @staticmethod
    def compute_in_channels(modalities, use_view_id=False):
        total = sum(Config.MODALITY_CHANNELS[m] for m in modalities)
        if use_view_id and 'view_id' not in modalities:
            total += Config.MODALITY_CHANNELS.get('view_id', 1)
        return total
