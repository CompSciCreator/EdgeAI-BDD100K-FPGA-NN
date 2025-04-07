import torch
from torch import nn
from step1_mm import test_dataloader
from step2_mm import ObjectDetectionNetwork  # Ensure correct import
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import ToPILImage
import torchvision.ops as ops  # For Non-Maximum Suppression

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping (matches Step3 training)
category_map = {
    'person': 0, 'rider': 1, 'car': 2, 'truck': 3, 'bus': 4,
    'train': 5, 'motorcycle': 6, 'bicycle': 7, 'traffic light': 8, 'traffic sign': 9
}
inverse_category_map = {v: k for k, v in category_map.items()}

# Model setup
model = ObjectDetectionNetwork(num_classes=10).to(device)
checkpoint_path = "model_epoch_10.pth"  # Adjust to desired epoch

# Load checkpoint
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")

model.eval()

# Inference function with visualization
def perform_inference_and_visualize(model, dataloader, confidence_threshold=0.5, num_images_to_show=3):
    predictions = []
    to_pil = ToPILImage()  # Convert tensor to PIL for plotting

    with torch.no_grad():
        for images, targets in dataloader:
            try:
                images = torch.stack(images).to(device)
            except:
                continue

            # Model prediction
            class_logits, box_logits = model(images)  # [batch, 10, 32, 32], [batch, 4, 32, 32]
            batch_size = images.size(0)

            # Apply softmax to class logits for probabilities
            class_probs = torch.softmax(class_logits, dim=1)  # [batch, 10, 32, 32]
            max_probs, pred_classes = class_probs.max(dim=1)  # [batch, 32, 32]

            # Process and visualize each image in the batch
            for b in range(min(batch_size, num_images_to_show)):
                img_preds = []
                boxes = []
                scores = []
                labels = []
                
                # Collect all predictions for this image
                for i in range(32):
                    for j in range(32):
                        prob = max_probs[b, i, j].item()
                        class_id = pred_classes[b, i, j].item()
                        
                        if prob < confidence_threshold or class_id == 9:
                            continue  # Skip low confidence and background
                        
                        # Get box coordinates (absolute in 256x256 space)
                        cx = box_logits[b, 0, i, j].item()
                        cy = box_logits[b, 1, i, j].item()
                        w = box_logits[b, 2, i, j].item()
                        h = box_logits[b, 3, i, j].item()
                        
                        # Convert to x1y1x2y2 format
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        
                        # Clamp coordinates to image bounds
                        x1, x2 = max(0, x1), min(256, x2)
                        y1, y2 = max(0, y1), min(256, y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue  # Skip invalid boxes
                        
                        boxes.append([x1, y1, x2, y2])
                        scores.append(prob)
                        labels.append(inverse_category_map[class_id])
                
                # Apply Non-Maximum Suppression
                if len(boxes) > 0:
                    boxes_tensor = torch.tensor(boxes)
                    scores_tensor = torch.tensor(scores)
                    
                    # Use NMS to filter boxes
                    keep_idx = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
                    
                    # Keep only surviving predictions
                    final_boxes = boxes_tensor[keep_idx].tolist()
                    final_scores = scores_tensor[keep_idx].tolist()
                    final_labels = [labels[i] for i in keep_idx]
                    
                    # Format into prediction list
                    img_preds = [
                        {'label': lbl, 'box': box, 'confidence': conf}
                        for box, lbl, conf in zip(final_boxes, final_labels, final_scores)
                    ]
                
                # Print textual predictions
                print(f"\nImage {b+1} Predictions:")
                if len(img_preds) == 0:
                    print("No objects detected")
                else:
                    for idx, pred in enumerate(img_preds, 1):
                        print(f"Object {idx}: {pred['label']}")
                        print(f"  Box: {[round(coord, 1) for coord in pred['box']]}")
                        print(f"  Confidence: {pred['confidence']:.2f}")

                # Convert image tensor to PIL for plotting
                img = to_pil(images[b].cpu())

                # Plotting
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(img)

                # Plot ground truth boxes (scaled to 256x256)
                gt_boxes = targets[b]['boxes']
                gt_labels = targets[b]['labels']
                for box, label in zip(gt_boxes, gt_labels):
                    x1, y1, x2, y2 = box
                    # Handle scaling if necessary
                    if x1 > 256 or x2 > 256 or y1 > 256 or y2 > 256:
                        original_width, original_height = targets[b]['original_size']
                        x1 = x1 * 256 / original_width
                        x2 = x2 * 256 / original_width
                        y1 = y1 * 256 / original_height
                        y2 = y2 * 256 / original_height
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, label, color='green', fontsize=10, 
                            bbox=dict(facecolor='white', alpha=0.5))

                # Plot predicted boxes
                for pred in img_preds:
                    x1, y1, x2, y2 = pred['box']
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x1, y1 - 5, f"{pred['label']} ({pred['confidence']:.2f})",
                        color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5)
                    )

                ax.set_title(f"Image {len(predictions) + 1}: Ground Truth (Green) vs Predicted (Red)")
                plt.axis('off')
                plt.show()

                predictions.append({
                    'image_idx': b,
                    'predictions': img_preds,
                    'ground_truth': targets[b]
                })

            break  # Process only one batch for visualization

    return predictions

# Run inference and visualize
print("Running inference and visualization on test set...")
results = perform_inference_and_visualize(model, test_dataloader, confidence_threshold=0.5, num_images_to_show=3)

print(f"\nTotal images processed: {len(results)}")