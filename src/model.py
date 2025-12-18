import segmentation_models_pytorch as smp
try:
    from config import Config
except ImportError:
    from .config import Config

def get_model():
    # Unet++ is generally better for defect detection than standard Unet
    # We enable Deep Supervision (from U-Net++ paper) for better gradient flow
    model = smp.UnetPlusPlus(
        encoder_name=Config.ENCODER, 
        encoder_weights=Config.ENCODER_WEIGHTS, 
        in_channels=Config.IN_CHANNELS,
        classes=len(Config.CLASSES), 
        activation=Config.ACTIVATION,
        # deep_supervision=True # Enabling this requires changing the training loop to handle list output
    )
    return model
