# This script generates a YAML file with the best hyperparameters for YOLOv8 model training.
# It performs the analysis of the dataset to find the best hyperparameters in terms of Class Weight Balancing based on object class distribution and anchor size based on bounding box dimensions using  k-means clustering in the prepared data subset.


import os
import numpy as np
from scipy.cluster.vq import kmeans
from tqdm import tqdm
import yaml

class YOLOHyperparamAnalyzer:
    def __init__(self, yolo_dir):
        self.yolo_dir = yolo_dir
        with open(os.path.join(yolo_dir, 'bdd100k.yaml')) as f:
            config = yaml.safe_load(f)
        self.class_names = config['names']
        
    def analyze_labels(self, split='train'):
        """Analyze labels from YOLO format files"""
        label_dir = os.path.join(self.yolo_dir, 'labels', split)
        class_counts = np.zeros(len(self.class_names), dtype=int)
        bbox_dimensions = []
        
        for label_file in tqdm(os.listdir(label_dir), desc=f"Analyzing {split}"):
            if not label_file.endswith('.txt'):
                continue
                
            with open(os.path.join(label_dir, label_file)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:  # YOLO format: class x_center y_center width height
                        class_id = int(parts[0])
                        w, h = map(float, parts[3:5])
                        
                        class_counts[class_id] += 1
                        bbox_dimensions.append([w, h])
        
        return class_counts, np.array(bbox_dimensions)
    
    def generate_hyperparameters(self):
        # Analyze both splits but weight towards training data
        train_counts, train_dims = self.analyze_labels('train')
        val_counts, val_dims = self.analyze_labels('val')
        
        # Combined statistics (70% train, 30% val)
        total_counts = train_counts + val_counts
        all_dims = np.concatenate([train_dims, val_dims])
        
        # Calculate weights (with smoothing)
        weights = 1 / np.sqrt((total_counts + 1) / (total_counts.max() + 1))
        weights = weights / weights.mean()
        
        # Anchor calculation (on training data only)
        anchors = self.calculate_anchors(train_dims)
        
        return {
            'path': self.yolo_dir,
            'train': 'images/train',
            'val': 'images/val',
            'names': self.class_names,
            'nc': len(self.class_names),
            'hyp': {
                'lr0': 0.01,
                'lrf': 0.01,
                'anchors': anchors,
                'fl_gamma': 2.0 if (total_counts.max() / total_counts.min()) > 10 else 1.5,
                'class_weights': weights.tolist(),
                'mosaic': 1.0,
                'mixup': 0.1,
                'copy_paste': 0.1
            }
        }
    
    def calculate_anchors(self, dims):
        """Calculate anchors for 640x640 input"""
        scaled = dims * 640
        anchors, _ = kmeans(scaled, 9)
        return anchors[np.argsort(anchors.prod(1))].tolist()

def main():
    YOLO_DIR = "./bdd100k_yolo_640"  # Your created subset
    analyzer = YOLOHyperparamAnalyzer(YOLO_DIR)
    config = analyzer.generate_hyperparameters()
    
    # Save updated config
    with open(os.path.join(YOLO_DIR, 'bdd100k_optimized.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"Generated optimized config at: {os.path.join(YOLO_DIR, 'bdd100k_optimized.yaml')}")

if __name__ == "__main__":
    main()