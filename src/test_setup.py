import torch
import segmentation_models_pytorch as smp
from config import Config
from dataset import DefectDataset, get_training_augmentation, get_preprocessing
from model import get_model

def test():
    print("Testing setup...")
    
    # 1. Dataset
    print("Initializing dataset...")
    preprocessing_fn = smp.encoders.get_preprocessing_fn(Config.ENCODER, Config.ENCODER_WEIGHTS)
    
    dataset = DefectDataset(
        Config.TRAIN_IMG_DIR, 
        Config.TRAIN_MASK_DIR, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        print("Iterating over all items...")
        for i in range(len(dataset)):
            try:
                image, mask = dataset[i]
                print(f"Item {i}: OK")
            except Exception as e:
                print(f"Item {i}: Failed - {e}")
        
        # 2. Model
        print("Initializing model...")
        model = get_model()
        model.to(Config.DEVICE)
        
        # 3. Forward
        print("Running forward pass...")
        input_tensor = image.unsqueeze(0).to(Config.DEVICE)
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")
        
        print("Test passed!")
    else:
        print("Dataset is empty.")

if __name__ == "__main__":
    test()
