# import torch
# from torch import nn, optim
# from step1_mm import train_loader, test_loader
# from step2_mm import model, device
# from datetime import datetime

# # Loss functions and optimizer
# cls_criterion = nn.CrossEntropyLoss()
# box_criterion = nn.SmoothL1Loss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)  # From 0.001 to 0.0001

# def train_epoch(loader, model, optimizer, epoch_num):
#     model.train()
#     total_loss = 0.0
#     print(f"\nEpoch {epoch_num} Started: {datetime.now().strftime('%H:%M:%S')}")
#     print(f"Total batches: {len(loader)} | Batch size: {loader.batch_size}")

#     for batch_idx, (images, targets) in enumerate(loader):
#         # Move data to device
#         images = images.to(device)
        
#         # Forward pass
#         class_logits, box_preds = model(images)
        
#         # Initialize losses
#         batch_cls_loss = 0.0
#         batch_box_loss = 0.0
#         valid_samples = 0
        
#         # Process each image in the batch
#         for i in range(len(images)):
#             # Get ground truth for this image
#             gt_boxes = targets[i]['boxes'].to(device)
#             gt_labels = targets[i]['labels'].to(device)
#             num_objects = gt_labels.shape[0]
            
#             if num_objects == 0:
#                 continue  # Skip images with no annotations
                
#             valid_samples += 1
            
#             # Reshape predictions: [C, H, W] -> [H*W, C]
#             pred_classes = class_logits[i].permute(1, 2, 0).reshape(-1, 10)
#             pred_boxes = box_preds[i].permute(1, 2, 0).reshape(-1, 4)
            
#             # Use first N predictions to match ground truth
#             cls_loss = cls_criterion(pred_classes[:num_objects], gt_labels)
#             # Modify box loss calculation
#             box_loss = box_criterion(pred_boxes[:num_objects], gt_boxes) * 0.1  # Scale down box loss

#             batch_cls_loss += cls_loss
#             batch_box_loss += box_loss

#         # Handle batches with no valid samples
#         if valid_samples == 0:
#             continue
            
#         # Average losses
#         avg_cls_loss = batch_cls_loss / valid_samples
#         avg_box_loss = batch_box_loss / valid_samples
#         total_batch_loss = avg_cls_loss + avg_box_loss
        
#         # Backpropagation
#         optimizer.zero_grad()
#         total_batch_loss.backward()
#         optimizer.step()
        
#         # Progress monitoring
#         total_loss += total_batch_loss.item()
#         if batch_idx % 50 == 0:
#             print(f"Batch {batch_idx+1}/{len(loader)} | "
#                     f"Loss: {total_batch_loss.item():.3f} | "
#                     f"CLS: {avg_cls_loss.item():.3f} | "
#                     f"BOX: {avg_box_loss.item():.3f}")

#     return total_loss / len(loader)

# if __name__ == "__main__":
#     print("====== Training Initialization ======")
#     print(f"Training on: {device}")
#     print(f"Training samples: {len(train_loader.dataset)}")
#     print(f"Model architecture:\n{model}")
    
#     start_time = datetime.now()
    
#     try:
#         for epoch in range(1, 11):
#             epoch_loss = train_epoch(train_loader, model, optimizer, epoch)
#             print(f"\nEpoch {epoch} Summary:")
#             print(f"Average Loss: {epoch_loss:.4f}")
#             print(f"Elapsed Time: {datetime.now() - start_time}")
            
#             # Save checkpoint
#             torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
#             print(f"Checkpoint saved: model_epoch_{epoch}.pth")
            
#     except KeyboardInterrupt:
#         print("\nTraining interrupted! Saving final model...")
#         torch.save(model.state_dict(), "model_interrupted.pth")
    
#     print(f"\nTotal training time: {datetime.now() - start_time}")
    
    
    
    
    
    
    # Title: Step3.py
# Description: This script trains the ObjectDetectionNetwork for the BDD100K dataset.
# It loads the dataset from step1_mm.py using train_dataloader and test_dataloader,
# imports the model from step2.py, and sets up training with classification and regression losses.
# Author: Andrew Paolella

import torch
from torch import nn, optim
from step1_mm import train_dataloader, test_dataloader
from step2_mm import ObjectDetectionNetwork
from datetime import datetime

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping (BDD100K's 10 classes)
category_map = {
    'person': 0, 'rider': 1, 'car': 2, 'truck': 3, 'bus': 4,
    'train': 5, 'motorcycle': 6, 'bicycle': 7, 'traffic light': 8, 'traffic sign': 9
}

# Model setup
model = ObjectDetectionNetwork(num_classes=10).to(device)
criterion_class = nn.CrossEntropyLoss()
criterion_box = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Logging
log_file = open("step3_10epoch.txt", "w")
start_time = datetime.now()

def log(msg):
    print(msg)
    log_file.write(msg + "\n")

def assign_targets_to_grid(targets, grid_size=32, image_size=256):
    """Assign ground truth boxes to grid cells based on center point."""
    batch_size = len(targets)
    grid = [{} for _ in range(batch_size)]
    cell_size = image_size / grid_size  # 256 / 32 = 8 pixels per cell

    for b in range(batch_size):
        for box, label in zip(targets[b]['boxes'], targets[b]['labels']):
            # Convert box to center_x, center_y, width, height
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            # Find grid cell
            grid_x = int(center_x / cell_size)
            grid_y = int(center_y / cell_size)

            # Clamp to grid bounds
            grid_x = max(0, min(grid_x, grid_size - 1))
            grid_y = max(0, min(grid_y, grid_size - 1))

            # Assign first object per cell (simplified approach)
            if (grid_y, grid_x) not in grid[b]:
                grid[b][(grid_y, grid_x)] = {
                    'label': category_map.get(label, 9),  # Default to last class if unknown
                    'box': torch.tensor([center_x, center_y, width, height], dtype=torch.float32)
                }

    return grid

def train_eval(dataloader, model, criterion_class, criterion_box, optimizer=None):
    model.train() if optimizer else model.eval()
    total_class_loss = 0
    total_box_loss = 0
    correct = 0
    total = 0

    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in dataloader:
            try:
                images = torch.stack(images).to(device)
            except:
                continue

            # Assign targets to grid
            grid_targets = assign_targets_to_grid(targets)

            if optimizer:
                optimizer.zero_grad()

            # Model prediction
            class_logits, box_logits = model(images)  # [batch, 10, 32, 32], [batch, 4, 32, 32]
            batch_size = images.size(0)

            # Flatten predictions for loss computation
            class_logits_flat = class_logits.view(batch_size * 32 * 32, 10)
            box_logits_flat = box_logits.view(batch_size * 32 * 32, 4)

            # Prepare ground truth tensors
            gt_classes = torch.full((batch_size * 32 * 32,), 9, dtype=torch.long, device=device)  # Default to background
            gt_boxes = torch.zeros(batch_size * 32 * 32, 4, device=device)
            mask = torch.zeros(batch_size * 32 * 32, dtype=torch.bool, device=device)

            # Fill ground truth for cells with objects
            for b in range(batch_size):
                for (grid_y, grid_x), target in grid_targets[b].items():
                    idx = b * 32 * 32 + grid_y * 32 + grid_x
                    gt_classes[idx] = target['label']
                    gt_boxes[idx] = target['box'].to(device)
                    mask[idx] = 1

            # Compute losses
            class_loss = criterion_class(class_logits_flat[mask], gt_classes[mask])
            box_loss = criterion_box(box_logits_flat[mask], gt_boxes[mask])

            total_loss = class_loss + box_loss

            if optimizer:
                total_loss.backward()
                optimizer.step()

            total_class_loss += class_loss.item()
            total_box_loss += box_loss.item()
            correct += (class_logits_flat[mask].argmax(1) == gt_classes[mask]).sum().item()
            total += mask.sum().item()

    avg_class_loss = total_class_loss / len(dataloader)
    avg_box_loss = total_box_loss / len(dataloader)
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy, avg_class_loss, avg_box_loss

# Training loop
try:
    epochs = 40
    for epoch in range(1, epochs + 1):
        train_acc, train_class_loss, train_box_loss = train_eval(
            train_dataloader, model, criterion_class, criterion_box, optimizer
        )
        log(f"Epoch {epoch}")
        log(f"Accuracy: {train_acc:.4f}")
        log(f"Avg class loss: {train_class_loss:.6f}")
        log(f"Avg box loss: {train_box_loss:.6f}\n")
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
        print(f"Checkpoint saved: model_epoch_{epoch}.pth")

except KeyboardInterrupt:
    log("\nTraining manually stopped.")

end_time = datetime.now()
log(f"Total time: {end_time - start_time}")
log_file.close()