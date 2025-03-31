import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import nms
from step1_mm import test_loader, CLASS_TO_IDX
from step2_mm import ObjectDetectionNetwork, device
import os

# Reverse class mapping
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

def load_model(model_path):
    model = ObjectDetectionNetwork().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        print(f"Loaded model from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")

def decode_boxes(box_output, grid_y, grid_x, stride=8, image_size=256):
    """Convert [x_center, y_center, w, h] relative to grid cell to [x1, y1, x2, y2] in image space"""
    boxes = box_output.clone()
    
    # Extract components
    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # Scale center coordinates to grid cell space (in pixels)
    x_center = x_center * stride
    y_center = y_center * stride
    
    # Scale width and height to grid cell space (not the entire image)
    w = w * stride
    h = h * stride
    
    # Add grid cell offsets (top-left corner of each grid cell)
    x_center = x_center + (grid_x * stride)
    y_center = y_center + (grid_y * stride)
    
    # Convert to [x1, y1, x2, y2]
    boxes[:, 0] = x_center - w / 2  # x1
    boxes[:, 1] = y_center - h / 2  # y1
    boxes[:, 2] = x_center + w / 2  # x2
    boxes[:, 3] = y_center + h / 2  # y2
    
    # Clamp to image boundaries
    return boxes.clamp(0, image_size)

def infer(model, loader, confidence_thresh=0.5, iou_thresh=0.3, image_size=256):
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            
            # Forward pass
            class_logits, box_preds = model(images)  # [B, C, H, W], [B, 4, H, W]
            
            batch_size, num_classes, grid_h, grid_w = class_logits.shape
            class_probs = torch.softmax(class_logits, dim=1)  # [B, C, H, W]
            
            # Calculate stride (pixels per grid cell)
            stride = image_size // grid_h  # Should be 256 / 32 = 8
            
            for i in range(batch_size):
                # Get max probability and class for each grid cell
                max_probs, class_ids = torch.max(class_probs[i], dim=0)  # [H, W]
                
                # Flatten predictions
                max_probs = max_probs.view(-1)  # [H*W]
                class_ids = class_ids.view(-1)  # [H*W]
                boxes = box_preds[i].permute(1, 2, 0).view(-1, 4)  # [H*W, 4]
                
                # Generate grid coordinates
                grid_x, grid_y = torch.meshgrid(torch.arange(grid_w), torch.arange(grid_h), indexing='xy')
                grid_x = grid_x.to(device).view(-1).float()
                grid_y = grid_y.to(device).view(-1).float()
                
                # Decode boxes
                boxes = decode_boxes(boxes, grid_y, grid_x, stride=stride, image_size=image_size)
                
                # Filter by confidence
                mask = max_probs > confidence_thresh
                filtered_boxes = boxes[mask]
                filtered_probs = max_probs[mask]
                filtered_classes = class_ids[mask]
                
                if len(filtered_boxes) == 0:
                    all_predictions.append({'boxes': [], 'scores': [], 'classes': []})
                    continue
                
                # Apply NMS
                keep = nms(filtered_boxes, filtered_probs, iou_thresh)
                
                all_predictions.append({
                    'boxes': filtered_boxes[keep].cpu().numpy(),
                    'scores': filtered_probs[keep].cpu().numpy(),
                    'classes': [IDX_TO_CLASS[c.item()] for c in filtered_classes[keep]]
                })
                
    return all_predictions

def plot_predictions(image, predictions, save_path=None):
    """Plot image with bounding boxes and labels"""
    plt.figure(figsize=(12, 8))
    image = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    
    ax = plt.gca()
    
    for box, score, cls in zip(predictions['boxes'], predictions['scores'], predictions['classes']):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Skip invalid boxes
        if width <= 0 or height <= 0:
            print(f"Skipping invalid box: {box}")
            continue
        
        rect = plt.Rectangle(
            (x1, y1), width, height,
            fill=False, linewidth=2,
            edgecolor='red', alpha=0.8
        )
        ax.add_patch(rect)
        
        label = f"{cls}: {score:.2f}"
        ax.text(
            x1, y1 - 5, label,
            color='red', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "model_epoch_10.pth"
    SAVE_DIR = "inference_results"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Run inference
    predictions = infer(model, test_loader)
    
    # Visualize first 5 samples
    for idx in range(5):
        image, _ = test_loader.dataset[idx]
        pred = predictions[idx]
        save_path = os.path.join(SAVE_DIR, f"result_{idx}.png")
        plot_predictions(image, pred, save_path)
        print(f"Saved visualization: {save_path}")