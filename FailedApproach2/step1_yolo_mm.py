import torch
import numpy as np
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BDD100KDetectionDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        self.data = []
        
        # Load label data
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        print(f"\nInitial annotations: {len(self.labels)}")
        
        # Image validation
        self.image_paths = {}
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    key = os.path.basename(file)
                    self.image_paths[key] = os.path.join(root, file)
        print(f"Discovered images: {len(self.image_paths)}")
        
        # Filter annotations to valid image-label pairs                
        for item in self.labels:
            if os.path.basename(item['name']) in self.image_paths:
                annotations = item.get('labels', [])
                self.data.append((self.image_paths[os.path.basename(item['name'])], annotations))
        print(f"Valid annotations after filtering: {len(self.data)}")
        
        if len(self.data) == 0:
            raise RuntimeError("No valid images+annotations pairs found.")
            
        # Class mapping to integer
        self.CLASSES = [
            'pedestrian', 'rider', 'car', 'truck', 'bus', 
            'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
        ]
        self.label_map = {name: idx for idx, name in enumerate(self.CLASSES)}
        self.idx_to_class = {idx: name for name, idx in self.label_map.items()}
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, annotations = self.data[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        
        if self.transform:
            image = self.transform(image)

        # Prepare YOLO format targets ( box2d format to YOLO format i.e. x1, y1, x2, y2 to cx, cy, w, h)
        targets = []
        for ann in annotations:
            if 'box2d' in ann and ann['category'] in self.label_map:
                bbox = ann['box2d']
                class_id = self.label_map[ann['category']]
                
                # Convert to YOLO format (normalized cx, cy, w, h)
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                cx = (x1 + x2) / 2 / original_width
                cy = (y1 + y2) / 2 / original_height
                w = (x2 - x1) / original_width
                h = (y2 - y1) / original_height
                
                targets.append([class_id, cx, cy, w, h])
        
        # Convert to tensor
        if len(targets) == 0:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        else:
            targets = torch.tensor(targets, dtype=torch.float32)
            
        return image, targets

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return torch.stack(images), targets

# Paths
TRAIN_DIR = "C://Users//User1//EdgeAI-BDD100K-FPGA-NN//archive//bdd100k//bdd100k//images//100k//train"
VALIDATION_DIR = "C://Users//User1//EdgeAI-BDD100K-FPGA-NN//archive//bdd100k//bdd100k//images//100k//val"
TRAIN_LABEL_PATH = "C://Users//User1//EdgeAI-BDD100K-FPGA-NN//archive//labels//det_v2_train_release.json"
VALIDATION_LABEL_PATH = "C://Users//User1//EdgeAI-BDD100K-FPGA-NN//archive//labels//det_v2_val_release.json"

batch_size = 8

# Create datasets
train_dataset = BDD100KDetectionDataset(TRAIN_DIR, TRAIN_LABEL_PATH, target_size=(256, 256))
val_dataset = BDD100KDetectionDataset(VALIDATION_DIR, VALIDATION_LABEL_PATH, target_size=(256, 256))

# Create subsets for training and validation Small subset
# (for demonstration purposes)
train_subset = Subset(train_dataset, indices=range(2000))
val_subset = Subset(val_dataset, indices=range(1000))

# Create DataLoaders
train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Print information
print(f"Total samples in training set: {len(train_dataset)}")
print(f"Total samples in test set: {len(val_dataset)}")
print(f"Unique labels: {len(train_dataset.CLASSES)}")
print(f"Labels: {train_dataset.CLASSES}")

print(f"\nTotal samples used in training subset: {len(train_subset)}")
print(f"Total samples used in test subset: {len(val_subset)}")



# Function to display a sample image with bounding boxes to visualize the dataset
def show_sample_with_boxes(image, targets, idx_to_class):
    # Denormalize image
    image = image.clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    # Convert targets to numpy if they're tensors
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()

    # Draw bounding boxes
    for t in targets:
        if len(t) >= 5:  # class_id, cx, cy, w, h
            class_id, cx, cy, w, h = t[:5]
            x1 = (cx - w/2) * image.shape[1]  # Convert to absolute coordinates
            y1 = (cy - h/2) * image.shape[0]
            width = w * image.shape[1]
            height = h * image.shape[0]
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 5, idx_to_class[int(class_id)],
                color='white', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5)
            )

    plt.axis("off")
    plt.show()


# # Get first batch & display sample
# Get first batch & display sample
for X, y in train_dataloader:
    print(f"\nBatch information:")
    print(f"Number of images in batch: {len(X)}")
    print(f"Shape of processed image tensor [C,H,W]: {X[0].shape}")
    
    # Display ground truths for first image
    print(f"\nGround truths for first image:")
    
    # Get first image's targets (y[0] is already in YOLO format: [num_objects, 5])
    targets = y[0].cpu().numpy()  # Convert to numpy for easier handling
    
    print(f"Number of objects: {len(targets)}")
    print("Bounding boxes (cx, cy, w, h in normalized coordinates) YOLO Format:")
    
    for i, t in enumerate(targets):
        class_id, cx, cy, w, h = t
        print(f"Object {i+1}:")
        print(f"  Category: {train_dataset.idx_to_class[int(class_id)]}")  # Use train_dataset instead of train_subset
        print(f"  Center: ({cx:.4f}, {cy:.4f})")
        print(f"  Width: {w:.4f}, Height: {h:.4f}")
        
        # Convert to x1,y1,x2,y2 for visualization if needed
        x1 = (cx - w/2) * X[0].shape[2]  # Multiply by image width
        y1 = (cy - h/2) * X[0].shape[1]  # Multiply by image height
        x2 = (cx + w/2) * X[0].shape[2]
        y2 = (cy + h/2) * X[0].shape[1]
        print(f"  Absolute coordinates: [x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}]")
    
    # Show first image with bounding boxes
    show_sample_with_boxes(X[0], y[0], train_dataset.idx_to_class)
    break