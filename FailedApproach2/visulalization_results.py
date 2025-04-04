import torch
import torchvision.ops as ops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from step2_yolo_mm import YOLOv8ResNet50
from step1_yolo_mm import val_dataloader

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("detection_results", exist_ok=True)

# Load trained model
model = YOLOv8ResNet50(num_classes=10).to(device)
checkpoint_path = "model_checkpoint_epoch_45.pth"

# Load checkpoint
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")

model.eval()

# Class names
CLASS_NAMES = [
    'pedestrian', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]

def decode_predictions(predictions, conf_thresh=0.5, iou_thresh=0.2, img_size=256):
    """Convert raw model outputs to detection results in YOLO format (cx, cy, w, h, conf, cls_id)"""
    detections = []
    # Use the anchor sizes as defined for 256x256 (since model was trained on 256x256)
    anchors = [
        [[10,13], [16,30], [33,23]],    # P3/8
        [[30,61], [62,45], [59,119]],   # P4/16
        [[116,90], [156,198], [373,326]] # P5/32
    ]
    # Heuristic scaling factor to increase predicted box sizes
    heuristic_scale = 4.0
    
    for scale_idx, pred in enumerate(predictions):
        grid_size = pred.shape[2]
        stride = img_size // grid_size
        
        pred = pred.view(pred.size(0), 3, 5 + len(CLASS_NAMES), grid_size, grid_size)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        pred[..., 4:] = torch.sigmoid(pred[..., 4:])
        
        for b in range(pred.size(0)):
            batch_detections = []
            for a in range(3):
                for gy in range(grid_size):
                    for gx in range(grid_size):
                        p = pred[b, a, gy, gx]
                        conf = p[4].item()
                        if conf < conf_thresh:
                            continue
                            
                        cls_conf, cls_id = torch.max(p[5:], 0)
                        total_conf = conf * cls_conf.item()
                        
                        # YOLO format: normalized cx, cy, w, h
                        cx = (gx + p[0].item()) / grid_size
                        cy = (gy + p[1].item()) / grid_size
                        # Scale w and h with a heuristic factor to increase box sizes
                        w = (torch.exp(p[2]) * anchors[scale_idx][a][0]) / img_size * heuristic_scale
                        h = (torch.exp(p[3]) * anchors[scale_idx][a][1]) / img_size * heuristic_scale
                        
                        batch_detections.append([
                            cx, cy, w.item(), h.item(), total_conf, int(cls_id.item())
                        ])
            
            if batch_detections:
                batch_detections = torch.tensor(batch_detections)
                keep = ops.nms(
                    torch.tensor([
                        [
                            det[0] - det[2]/2,
                            det[1] - det[3]/2,
                            det[0] + det[2]/2,
                            det[1] + det[3]/2
                        ] for det in batch_detections
                    ]),
                    batch_detections[:, 4],
                    iou_thresh
                )
                detections.append(batch_detections[keep])
            else:
                detections.append(torch.tensor([]))
    
    return detections

def save_detection_results(image, targets, predictions, filename, img_size=256):
    """Visualization with ground truth and predictions side by side"""
    # Denormalize image
    image = image.cpu().clone()
    for t, m, s in zip(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    
    # Ground Truth subplot
    ax1.imshow(image)
    ax1.set_xlim(0, img_size)
    ax1.set_ylim(img_size, 0)  # Matplotlib origin is top-left
    ax1.axis('off')
    ax1.set_title("Ground Truth")
    
    # Predictions subplot
    ax2.imshow(image)
    ax2.set_xlim(0, img_size)
    ax2.set_ylim(img_size, 0)
    ax2.axis('off')
    ax2.set_title("Predictions")

    def draw_box(ax, cx, cy, w, h, label, color):
        """Draw box and place label within image boundaries"""
        x1 = (cx - w/2) * img_size
        y1 = (cy - h/2) * img_size
        width = w * img_size
        height = h * img_size
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=1.5, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Adjust label position to stay within image
        label_y = y1 - 5
        if label_y < 0:  # If label would be above image, place it below
            label_y = y1 + height + 15
        ax.text(
            x1, label_y, label,
            color='white', fontsize=8,
            bbox=dict(facecolor=color, alpha=0.7, pad=0, edgecolor='none')
        )

    # Draw ground truth (green) on first subplot
    if targets is not None:
        for t in targets:
            if len(t) >= 5:  # class_id, cx, cy, w, h
                cls_id, cx, cy, w, h = t[:5]
                print("Ground Truth w, h:", w.item(), h.item())  # Debug print
                draw_box(
                    ax1, cx, cy, w, h,
                    f'GT: {CLASS_NAMES[int(cls_id)]}',
                    'lime'
                )

    # Draw predictions (red) on second subplot with confidence scores
    if predictions is not None and len(predictions) > 0:
        for det in predictions:
            if isinstance(det, torch.Tensor):
                det = det.cpu().numpy()
            
            # Predictions are in YOLO format: [cx, cy, w, h, conf, cls_id]
            cx, cy, w, h, conf, cls_id = det
            print("Predicted w, h:", w, h)  # Debug print
            draw_box(
                ax2, cx, cy, w, h,
                f'{CLASS_NAMES[int(cls_id)]} {conf:.2f}',
                'red'
            )

    plt.savefig(
        f"detection_results/{filename}.png",
        bbox_inches='tight',
        pad_inches=0,
        dpi=100
    )
    plt.close()

# Process and save 5 sample images
num_samples = 5
for i, (images, targets) in enumerate(val_dataloader):
    if i >= num_samples:
        break
    
    images = images.to(device)
    with torch.no_grad():
        preds = model(images)
        detections = decode_predictions(preds)
    
    # Convert tensor targets to list of arrays
    targets_list = [t.cpu().numpy() for t in targets]
    
    # Save results for the first image in the batch
    save_detection_results(images[0], targets_list[0], detections[0], f"sample_{i}")
    print(f"Saved results for sample {i}")

print(f"Saved {num_samples} detection visualizations to 'detection_results' folder")