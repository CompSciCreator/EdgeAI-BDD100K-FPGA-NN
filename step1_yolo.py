    
# This script converts BDD100K dataset to YOLO format
# and visualizes the converted dataset with bounding boxes
# and saves the converted dataset to a specified directory

#(DIRECTORY)>> bdd100k_yolo_640/
            # ├── images/
            # │   ├── train/      # Training images
            # │   └── val/       # Validation images
            # └── labels/
            #     ├── train/      # Training labels in YOLO format
            #     └── val/        # Validation labels in YOLO format
            # └── bdd100k.yaml    # Dataset configuration file for YOLOv8

import os
import json
from PIL import Image
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

class BDD100KToYOLOConverter:
    def __init__(self):
        self.CLASSES = [
            'pedestrian', 'rider', 'car', 'truck', 'bus',
            'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
        ]
        self.label_map = {name: idx for idx, name in enumerate(self.CLASSES)}
        self.idx_to_class = {idx: name for name, idx in self.label_map.items()}
        self.samples_to_verify = []  # Store multiple samples for verification

    def validate_annotations(self, item):
        """Check if annotation item has valid labels"""
        if not isinstance(item, dict):
            return False
        if 'labels' not in item:
            return False
        return any('box2d' in ann and ann.get('category') in self.label_map 
                    for ann in item.get('labels', []) if isinstance(ann, dict))

    def process_dataset(self, image_dir, label_path, output_dir, split='train', max_samples=2000, num_verify_samples=3):
        """Process dataset split and collect verification samples"""
        # Create output directories
        output_img_dir = os.path.join(output_dir, 'images', split)
        output_label_dir = os.path.join(output_dir, 'labels', split)
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        # Load labels
        with open(label_path, 'r') as f:
            labels = json.load(f)

        processed_count = 0
        self.samples_to_verify = []  # Reset for each dataset split

        for item in tqdm(labels[:max_samples], desc=f"Processing {split}"):
            if not self.validate_annotations(item):
                continue

            img_name = os.path.basename(item['name'])
            img_path = os.path.join(image_dir, img_name)
            
            if not os.path.exists(img_path):
                continue

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    img_resized = img.resize((640, 640), Image.Resampling.LANCZOS)
                    
                    # Process annotations
                    annotations = []
                    for ann in item.get('labels', []):
                        if not isinstance(ann, dict):
                            continue
                        if 'box2d' in ann and ann.get('category') in self.label_map:
                            bbox = ann['box2d']
                            class_id = self.label_map[ann['category']]
                            
                            # Convert to YOLO format
                            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                            cx = (x1 + x2) / 2 / width
                            cy = (y1 + y2) / 2 / height
                            w = (x2 - x1) / width
                            h = (y2 - y1) / height
                            
                            if w <= 0 or h <= 0:  # Skip invalid boxes
                                continue
                                
                            annotations.append((class_id, cx, cy, w, h))

                    if not annotations:
                        continue

                    # Save files
                    output_img_path = os.path.join(output_img_dir, img_name)
                    img_resized.save(output_img_path)
                    
                    label_file = os.path.join(output_label_dir, img_name.replace('.jpg', '.txt'))
                    with open(label_file, 'w') as f:
                        for ann in annotations:
                            f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

                    # Store samples for verification
                    if len(self.samples_to_verify) < num_verify_samples:
                        self.samples_to_verify.append({
                            'image': img_resized,
                            'annotations': annotations,
                            'img_path': output_img_path,
                            'label_path': label_file
                        })

                    processed_count += 1
            except Exception as e:
                print(f"\nError processing {img_name}: {str(e)}")
                continue

        print(f"\n{processed_count} images processed for {split} split")
        return processed_count

    def verify_samples(self, num_samples=3):
        """Visualize multiple samples with annotations"""
        if not self.samples_to_verify:
            print("No samples available for verification")
            return

        print(f"\n{'='*40}\nVerifying {min(num_samples, len(self.samples_to_verify))} samples\n{'='*40}")
        
        for i, sample in enumerate(self.samples_to_verify[:num_samples], 1):
            plt.figure(figsize=(12, 12))
            ax = plt.gca()
            ax.imshow(sample['image'])
            
            # Draw bounding boxes
            for ann in sample['annotations']:
                class_id, cx, cy, w, h = ann
                x1 = (cx - w/2) * 640  # Convert to absolute coordinates
                y1 = (cy - h/2) * 640
                width = w * 640
                height = h * 640
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 5, self.idx_to_class[class_id],
                    color='white', fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.7)
                )
            
            plt.title(f"Sample {i}: {os.path.basename(sample['img_path'])}\n"
                        f"Labels: {os.path.basename(sample['label_path'])}")
            plt.axis('off')
            plt.show()
            
            # Print annotation details
            print(f"\nSample {i} Details:")
            print(f"Image: {sample['img_path']}")
            print(f"Labels: {sample['label_path']}")
            for j, ann in enumerate(sample['annotations'], 1):
                print(f"  Object {j}: {self.idx_to_class[ann[0]]} "
                        f"Center: ({ann[1]:.4f}, {ann[2]:.4f}) "
                        f"Size: {ann[3]:.4f}x{ann[4]:.4f}")

def main():
    converter = BDD100KToYOLOConverter()
    
    # Path configuration
    BASE_DIR = "/Users/andrewpaolella/Desktop/Final-Project/BDD100k"
    TRAIN_DIR = os.path.join(BASE_DIR, "bdd100k", "bdd100k", "images", "100k", "train")
    VAL_DIR = os.path.join(BASE_DIR, "bdd100k", "bdd100k", "images", "100k", "val")
    TRAIN_LABEL_PATH = os.path.join(BASE_DIR, "labels", "det_v2_train_release.json")
    VAL_LABEL_PATH = os.path.join(BASE_DIR, "labels", "det_v2_val_release.json")
    OUTPUT_DIR = "./bdd100k_yolo_640"
    
    # Process datasets
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=== Processing Training Set ===")
    converter.process_dataset(TRAIN_DIR, TRAIN_LABEL_PATH, OUTPUT_DIR, split='train', max_samples=2000)
    
    print("\n=== Processing Validation Set ===")
    converter.process_dataset(VAL_DIR, VAL_LABEL_PATH, OUTPUT_DIR, split='val', max_samples=1000)
    
    # Create YAML config
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'names': converter.idx_to_class
    }
    config_path = os.path.join(OUTPUT_DIR, 'bdd100k.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    # Verify samples from both sets
    print("\n=== Verification ===")
    converter.verify_samples(num_samples=3)
    
    print("\n=== Dataset Preparation Complete ===")
    print(f"YOLO format dataset saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Configuration file: {config_path}")

if __name__ == "__main__":
    main()