import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from step1_yolo_mm import BDD100KDetectionDataset, collate_fn, train_dataloader, val_dataloader
from step2_yolo_mm import YOLOv8ResNet50
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import matplotlib.patches as patches

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)

# YOLOv8 loss function
# This loss function is designed for 256x256 input size
class YOLOv8Loss(nn.Module):
    def __init__(self, num_classes=10, input_size=256):  # Now explicitly using 256
        super(YOLOv8Loss, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        
        # Define base anchors (for 640x640 input) default YOLOv8 anchors
        # base_anchors = [
        #     [[10,13], [16,30], [33,23]],    # For small objects
        #     [[30,61], [62,45], [59,119]],   # For medium objects
        #     [[116,90], [156,198], [373,326]] # For large objects
        # ]
        
        # Scaled anchors for 256x256 (256/640 = 0.4 scale factor)
        self.anchors = torch.tensor([
            [[4,5], [6,12], [13,9]],      # P3/8 (small objects) - scaled by 0.4
            [[12,24], [25,18], [24,48]],   # P4/16 (medium objects)
            [[46,36], [62,79], [149,130]]  # P5/32 (large objects)
        ], device=device)
        
        # Note: These are pre-scaled for 256x256 input, so we don't need scale_factor anymore
        # The anchors are in absolute pixel values for 256x256 input
        
    def forward(self, predictions, targets):
        total_loss = 0
        box_loss = 0
        obj_loss = 0
        cls_loss = 0
        
        for scale_idx, pred in enumerate(predictions):
            
            # Get grid size and stride
            grid_size = pred.shape[2]
            stride = self.input_size // grid_size  # Using input_size (256) 
            
            # Reshape prediction 
            pred = pred.view(pred.size(0), 3, 5 + self.num_classes, grid_size, grid_size)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Initialize target tensors 
            target_tensor = torch.zeros_like(pred)
            
            # Process targets for this scale 
            for batch_idx, target in enumerate(targets):
                for t in target:
                    class_id, cx, cy, w, h = t
                    
                    # Find grid cell 
                    grid_x = int(cx * grid_size)
                    grid_y = int(cy * grid_size)
                    
                    # Find best anchor - now using pre-scaled anchors
                    anchor_ious = []
                    for anchor_idx in range(3):
                        # Get anchor dimensions in feature map space
                        anchor_w = self.anchors[scale_idx][anchor_idx][0] / stride
                        anchor_h = self.anchors[scale_idx][anchor_idx][1] / stride
                        iou = self._calculate_iou([0, 0, w, h], [0, 0, anchor_w, anchor_h])
                        anchor_ious.append(iou)
                    best_anchor = torch.argmax(torch.tensor(anchor_ious))
                    
                    # Find best anchor -  using pre-scaled anchors
                    anchor_ious = []
                    for anchor_idx in range(3):
                        # Get anchor dimensions in feature map space
                        anchor_w = self.anchors[scale_idx][anchor_idx][0] / stride
                        anchor_h = self.anchors[scale_idx][anchor_idx][1] / stride
                        iou = self._calculate_iou([0, 0, w, h], [0, 0, anchor_w, anchor_h])
                        anchor_ious.append(iou)
                    best_anchor = torch.argmax(torch.tensor(anchor_ious))
                    
                    
                    # Assign target
                    target_tensor[batch_idx, best_anchor, grid_y, grid_x, 0:4] = torch.tensor([cx, cy, w, h])
                    target_tensor[batch_idx, best_anchor, grid_y, grid_x, 4] = 1.0  # Objectness
                    target_tensor[batch_idx, best_anchor, grid_y, grid_x, 5 + int(class_id)] = 1.0
            
            # Calculate losses
            obj_mask = target_tensor[..., 4] == 1
            noobj_mask = target_tensor[..., 4] == 0
            
            # Box loss (only for cells with objects)
            box_pred = pred[..., 0:4][obj_mask]
            box_target = target_tensor[..., 0:4][obj_mask]
            box_loss += self.mse(box_pred, box_target)
            
            # Objectness loss
            obj_pred = pred[..., 4]
            obj_target = target_tensor[..., 4]
            obj_loss += self.bce(obj_pred[obj_mask], obj_target[obj_mask])
            obj_loss += 0.5 * self.bce(obj_pred[noobj_mask], obj_target[noobj_mask])
            
            # Classification loss (only for cells with objects)
            cls_pred = pred[..., 5:][obj_mask]
            cls_target = target_tensor[..., 5:][obj_mask]
            cls_loss += self.bce(cls_pred, cls_target)
        
        total_loss = box_loss + obj_loss + cls_loss
        return total_loss, box_loss.item(), obj_loss.item(), cls_loss.item()
    
    def _calculate_iou(self, box1, box2):
        # Calculate IoU between two boxes [x1,y1,w1,h1]
        box1 = torch.tensor([box1[0]-box1[2]/2, box1[1]-box1[3]/2, box1[0]+box1[2]/2, box1[1]+box1[3]/2])
        box2 = torch.tensor([box2[0]-box2[2]/2, box2[1]-box2[3]/2, box2[0]+box2[2]/2, box2[1]+box2[3]/2])
        
        # Intersection area
        inter_area = (torch.min(box1[2], box2[2]) - torch.max(box1[0], box2[0])) * \
                        (torch.min(box1[3], box2[3]) - torch.max(box1[1], box2[1]))
        inter_area = torch.clamp(inter_area, min=0)
        
        # Union area
        union_area = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter_area
        
        return inter_area / (union_area + 1e-6)

# Initialize model, loss, and optimizer
model = YOLOv8ResNet50(num_classes=10).to(device)
criterion = YOLOv8Loss(num_classes=10, input_size=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training setup
num_epochs = 1

train_loss_history = []
val_loss_history = []
box_loss_history = []
obj_loss_history = []
cls_loss_history = []
lr_history = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("training_plots", exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    epoch_box_loss = 0
    epoch_obj_loss = 0
    epoch_cls_loss = 0
    
    for batch_idx, (images, targets) in enumerate(train_dataloader):
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        loss, b_loss, o_loss, c_loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        epoch_train_loss += loss.item()
        epoch_box_loss += b_loss
        epoch_obj_loss += o_loss
        epoch_cls_loss += c_loss
        
        # Print batch statistics
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_dataloader)} | "
                    f"Loss: {loss.item():.4f} (Box: {b_loss:.4f}, Obj: {o_loss:.4f}, Cls: {c_loss:.4f})")
    
    # Update learning rate and record
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    # Calculate epoch averages
    epoch_train_loss /= len(train_dataloader)
    epoch_box_loss /= len(train_dataloader)
    epoch_obj_loss /= len(train_dataloader)
    epoch_cls_loss /= len(train_dataloader)
    
    # Store history
    train_loss_history.append(epoch_train_loss)
    box_loss_history.append(epoch_box_loss)
    obj_loss_history.append(epoch_obj_loss)
    cls_loss_history.append(epoch_cls_loss)
    
    # Save checkpoint after each epoch
    checkpoint_path = f"checkpoints/epoch_{epoch+1}_checkpoint.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'lr_history': lr_history,
    }, checkpoint_path)
    
    # Keep only the latest checkpoint by removing previous ones
    if epoch > 0:
        prev_checkpoint = f"checkpoints/epoch_{epoch}_checkpoint.pth"
        if os.path.exists(prev_checkpoint):
            os.remove(prev_checkpoint)
    
    # Validation
    if (epoch+1) % 5 == 0 or epoch == num_epochs - 1:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_dataloader:
                images = images.to(device)
                outputs = model(images)
                loss, _, _, _ = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        val_loss_history.append(val_loss)
        
        print(f"\nValidation Loss: {val_loss:.4f}")
        print("-" * 50)
    
    # Plot and save graphs every 5 epochs
    if (epoch+1) % 5 == 0 or epoch == num_epochs - 1:
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(train_loss_history, label='Train Loss')
        if val_loss_history:
            val_x = [5*(i+1) for i in range(len(val_loss_history))]
            plt.plot(val_x, val_loss_history, label='Val Loss', marker='o')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Component loss plot
        plt.subplot(1, 3, 2)
        plt.plot(box_loss_history, label='Box Loss')
        plt.plot(obj_loss_history, label='Obj Loss')
        plt.plot(cls_loss_history, label='Cls Loss')
        plt.title('Component Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(1, 3, 3)
        plt.plot(lr_history, label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.legend()
        plt.grid(True)
        
        
        plt.tight_layout()
        plt.savefig(f'training_plots/epoch_{epoch+1}_{timestamp}.png')
        plt.close()

# After training completes, rename the last checkpoint to final model
final_checkpoint = f"checkpoints/epoch_{num_epochs}_checkpoint.pth"
if os.path.exists(final_checkpoint):
    os.rename(final_checkpoint, "yolov8_resnet50_final.pth")

# Final plots
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot([5*(i+1) for i in range(len(val_loss_history))], val_loss_history, label='Val Loss', marker='o')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(box_loss_history, label='Box Loss')
plt.plot(obj_loss_history, label='Obj Loss')
plt.plot(cls_loss_history, label='Cls Loss')
plt.title('Component Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(lr_history)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.grid(True)

plt.tight_layout()
plt.savefig('final_training_metrics.png')
plt.show()

print("Training complete. Model and plots saved.")