#fine-tuned the hyperparameters based on the analysis from dataset
#To run this script, you need to have the YOLOv8 library installed. You can install it using pip:
#pip install ultralytics
#step1_YOLO.py prepares the dataset to YOLO format and 
# YOLO_hyperparam.py generates a YAML file with the best hyperparameters for the dataset. It performs the analysis of the dataset to find the best hyperparameters in terms of Class Weight Balancing based on object class distribution and anchor size based on bounding box dimensions using  k-means clustering in the prepared data subset. 
# step2_YOLO_update.py this script fine-tunes the YOLOv8 model using the generated YAML file with the best hyperparameters. It trains the model on the prepared dataset and saves the trained model. It also validates the model.

from ultralytics import YOLO
import torch
import os
import yaml
import cv2

# os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Minimal memory overhead
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    
    
    
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False  # Disable benchmark for determinism
torch.manual_seed(42)  # Also set random seeds for reproducibility

class YOLOv8FineTuner:
    def __init__(self, config_path):
        self.config_path = config_path
        self.device = self._get_device()
        self.setup_paths()
        
        
    def _get_device(self):
        """Automatically select GPU if available"""
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            return '0'  # Use first GPU
        # raise RuntimeError("No GPU available - required for training")
        print("Using CPU")
        return 'cpu'
        
    def setup_paths(self):
        """Load dataset config and hyperparameters"""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        self.data_dir = config['path']
        self.train_dir = os.path.join(self.data_dir, config['train'])
        self.val_dir = os.path.join(self.data_dir, config['val'])
        self.hyp = config.get('hyp', {})  # Load hyperparameters
        
        assert os.path.exists(self.train_dir), f"Train directory not found: {self.train_dir}"
        assert os.path.exists(self.val_dir), f"Val directory not found: {self.val_dir}"
        

    def prepare_model(self, model_name='yolov8n.pt', freeze_backbone=False):
        """Initialize model with optimized hyperparameters"""
        # Load model with hyperparameters from YAML
        self.model = YOLO(model_name)
        
        # Print loaded hyperparameters for verification
        if hasattr(self, 'hyp'):
            print("\nLoaded Hyperparameters:")
            print(f"- Anchors: {self.hyp.get('anchors', 'default')}")
            print(f"- Focal Gamma: {self.hyp.get('fl_gamma', 'default')}")
            print(f"- Class Loss Gain: {self.hyp.get('cls', 'default')}")
        
        if freeze_backbone:
            for param in self.model.model.parameters():
                param.requires_grad = False
            for param in self.model.model.model[-1].parameters():
                param.requires_grad = True
        
        return self.model
            

    def train(self, **kwargs):
        """Run fine-tuning with clear epoch settings"""
        # Get base arguments
        train_args = self.get_train_args()
        
        # Merge with any additional kwargs
        train_args.update(kwargs)
        
        # Device is already set in the model initialization
        if 'device' in train_args:
            del train_args['device']
        
        results = self.model.train(
            data=os.path.abspath(self.config_path),
            **train_args
        )
        return results

    def validate(self):
        """Run validation"""
        metrics = self.model.val()
        print(f"mAP50-95: {metrics.box.map:.4f}")
        return metrics

    # def export_for_edge(self, format='onnx', imgsz=(640, 640)):
    #     """Export model"""
    #     export_path = os.path.join(self.data_dir, f"yolov8_bdd100k.{format}")
    #     self.model.export(format=format, imgsz=imgsz, simplify=True, opset=12)
    #     print(f"Model exported to {export_path}")
    #     return export_path

    def save_model(self, path="yolov8_custom.pt"):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")

    def visualize_results(self, image_path):
        """Visualize with OpenCV"""
        results = self.model.predict(image_path)
        for r in results:
            img = cv2.imread(image_path)
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return results
    
    
    def get_train_args(self):
        """Generate training arguments"""
        args = {
            'epochs': 200,
            'imgsz': 640,
            'batch': 16,
            'optimizer': 'AdamW',
            'pretrained': False,
            'augment': True
        }
        
        if hasattr(self, 'hyp'):
            args.update({
                'lr0': self.hyp.get('lr0', 0.01),
                'mosaic': self.hyp.get('mosaic', 1.0),
                'mixup': self.hyp.get('mixup', 0.1),
                'fliplr': 0.5,
                'label_smoothing': 0.1,
                'amp': True if self.device == '0' else False
            })
        
        return args

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            return '0'  # Use GPU
    except Exception as e:
        print(f"GPU check failed: {e}")
    return 'cpu'  # Fallback to CPU


def main():
    CONFIG_PATH = os.path.abspath("./bdd100k_yolo_640/bdd100k_optimized.yaml")
    PRETRAINED_MODEL = "yolov8n.pt"
    
    finetuner = YOLOv8FineTuner(CONFIG_PATH)
    model = finetuner.prepare_model(PRETRAINED_MODEL)
    
    # # Start training
    # print("\nStarting training with parameters:")
    # for k, v in train_args.items():
    #     print(f"- {k}: {v}")
    # finetuner.train(**train_args)
    
    finetuner.train(**finetuner.get_train_args())
    
    # Save the trained model
    finetuner.save_model("trained_yolov8n.pt")
    
    # Validate and export
    print("\nValidation results:")
    finetuner.validate()
    
    # print("\nExporting model...")
    # finetuner.export_for_edge(format='onnx')
    
    # Visualize
    sample_image = os.path.join(finetuner.data_dir, "images/val/0000f77c-62c2a288.jpg")
    if os.path.exists(sample_image):
        print("\nSample inference:")
        finetuner.visualize_results(sample_image)

if __name__ == "__main__":
    main()