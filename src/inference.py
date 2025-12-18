import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
try:
    from config import Config
    from dataset import SomicDataset, get_validation_augmentation, get_preprocessing
    from model import get_model
except ImportError:
    from .config import Config
    from .dataset import SomicDataset, get_validation_augmentation, get_preprocessing
    from .model import get_model

def get_bounding_boxes(mask_class_2):
    """
    Find bounding boxes for the defect class (value 2).
    mask_class_2: numpy array (H, W) where 1 is defect, 0 is others.
    """
    contours, _ = cv2.findContours(mask_class_2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5: # Filter small noise
            boxes.append((x, y, x+w, y+h))
    return boxes

def visualize_multiclass(image, mask, prediction, save_path, boxes=None):
    """
    image: (6, H, W) numpy array
    mask: (H, W) numpy array (0, 1, 2)
    prediction: (H, W) numpy array (0, 1, 2)
    """
    
    # Extract Texture (first 3 channels)
    # Image is likely normalized. We need to denormalize for visualization.
    # Assuming simple min-max normalization was done or standard scaling.
    # For visualization, we just map min-max to 0-1.
    texture = image[:3, :, :].transpose(1, 2, 0) # (H, W, 3)
    
    if texture.max() > texture.min():
        texture = (texture - texture.min()) / (texture.max() - texture.min())
    
    # Convert to uint8 for opencv drawing
    texture_uint8 = (texture * 255).astype(np.uint8)
    # Dataset loads as RGB, so this is RGB.
    
    # Create a color map for masks
    # 0: Black (Background), 1: Gray (Part), 2: Red (Defect)
    def colorize_mask(m):
        h, w = m.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[m == 1] = [128, 128, 128] # Part - Gray
        color_mask[m == 2] = [255, 0, 0]     # Defect - Red
        return color_mask

    gt_color = colorize_mask(mask)
    pred_color = colorize_mask(prediction)
    
    # Draw boxes on prediction visualization
    pred_with_boxes = pred_color.copy()
    if boxes:
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(pred_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
            
    # Overlay on texture
    overlay = texture_uint8.copy()
    # Create mask for defect
    defect_mask = (prediction == 2).astype(np.uint8)
    
    # Apply red overlay for defects
    # We can blend it
    red_layer = np.zeros_like(overlay)
    red_layer[defect_mask == 1] = [255, 0, 0]
    
    # Blend where defect is present (skip if no pixels to avoid None from addWeighted)
    mask_bool = (defect_mask == 1)
    if mask_bool.any():
        overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 0.7, red_layer[mask_bool], 0.3, 0)
    
    # Draw boxes on overlay
    if boxes:
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(texture)
    plt.title('Input Texture')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(gt_color)
    plt.title('Ground Truth\n(Gray=Part, Red=Defect)')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(pred_with_boxes)
    plt.title('Prediction + Boxes')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title('Defect Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_inference():
    print("Loading model...")
    model = get_model()
    model_path = os.path.join(Config.MODEL_SAVE_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Load state dict
    state_dict = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE)
    model.eval()
    
    print("Loading validation dataset...")
    preprocessing_fn = smp.encoders.get_preprocessing_fn(Config.ENCODER, Config.ENCODER_WEIGHTS)
    
    # Use validation dataset
    valid_dataset = SomicDataset(
        Config.VAL_IMG_DIR, 
        Config.VAL_MASK_DIR, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=Config.CLASSES,
    )
    
    output_dir = 'inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running inference on {len(valid_dataset)} images...")
    
    for i in range(len(valid_dataset)):
        image_np, mask_np = valid_dataset[i]
        
        # Prepare input
        # Convert numpy to tensor
        x_tensor = torch.from_numpy(image_np).float().to(Config.DEVICE).unsqueeze(0) # (1, 6, H, W)
        
        with torch.no_grad():
            logits = model(x_tensor) # (1, 3, H, W)
            # Get class with highest probability
            pr_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy() # (H, W)
            
        # Get ground truth mask
        gt_mask = mask_np.squeeze() # (H, W)
        
        # Check GT defects
        gt_defects = np.sum(gt_mask == 2)
        print(f"Image {i}: GT Defect Pixels: {gt_defects}")
        
        # Extract bounding boxes for Defect (Class 2)
        defect_mask = (pr_mask == 2)
        pred_defects = np.sum(defect_mask)
        print(f"Image {i}: Pred Defect Pixels: {pred_defects}")
        
        boxes = get_bounding_boxes(defect_mask)
        
        if len(boxes) > 0:
            print(f"Image {i}: Found {len(boxes)} defects.")
            for idx, box in enumerate(boxes):
                print(f"  Defect {idx+1}: Box {box} (x1, y1, x2, y2)")
        else:
            print(f"Image {i}: No defects found.")
            
        # Visualize
        save_path = os.path.join(output_dir, f"result_{i}.png")
        visualize_multiclass(image_np, gt_mask, pr_mask, save_path, boxes)
        
    print(f"Inference complete. Results saved to {output_dir}")

if __name__ == '__main__':
    run_inference()
