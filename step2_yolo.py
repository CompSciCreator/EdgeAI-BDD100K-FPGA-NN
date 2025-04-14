# YOLOv8 Fine-Tuning Script
#loads the YOLOv8 model, prepares it for training, and fine-tunes it on a custom BDD100k dataset generated in step1. It also validates the model.


# from ultralytics import YOLO
# import torch

# def main():
#     # Load smallest YOLOv8 model (best for edge)
#     model = YOLO("yolov8n.pt") 
    
#     # Train with edge-friendly settings
#     results = model.train(
#         data="bdd100k.yaml",
#         imgsz=256,
#         epochs=50,
#         batch=32,  # Reduce if OOM
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         optimizer="AdamW",
#         lr0=0.01,
#         weight_decay=1e-4,
#         augment=True,  # Mosaic, flip, etc.
#         rect=False,    # Disable rectangular training for fixed 256x256
#         pretrained=True,
#         patience=10,   # Early stopping
#         verbose=True
#     )

#     # Validate
#     metrics = model.val()
#     print(f"mAP@0.5: {metrics.box.map}")  # Key metric

#     # Export for edge (TensorRT recommended)
#     model.export(
#         format="onnx",  # or "engine" for TensorRT
#         imgsz=(256, 256),
#         simplify=True,
#         opset=12
#     )

# if __name__ == "__main__":
#     main()

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
        """Verify dataset structure"""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        self.data_dir = config['path']
        self.train_dir = os.path.join(self.data_dir, config['train'])
        self.val_dir = os.path.join(self.data_dir, config['val'])
        
        assert os.path.exists(self.train_dir), f"Train directory not found: {self.train_dir}"
        assert os.path.exists(self.val_dir), f"Val directory not found: {self.val_dir}"

    def prepare_model(self, model_name='yolov8n.pt', freeze_backbone=False):
        """Load pretrained model with proper device handling"""
        self.model = YOLO(model_name)
        
        
        
        if freeze_backbone:
            for param in self.model.model.parameters():
                param.requires_grad = False
            for param in self.model.model.model[-1].parameters():
                param.requires_grad = True
            print("Backbone frozen - only training head layers")
        return self.model
    

    def train(self, epochs=100, imgsz=640, batch=16, lr0=0.01, **kwargs):
        """Run fine-tuning with clear epoch settings"""
        results = self.model.train(
            data=os.path.abspath(self.config_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            device=self.device,  # Use the detected device
            optimizer='AdamW',
            pretrained=False,  # Already handled by YOLO() initialization
            **kwargs
        )
        return results

    def validate(self):
        """Run validation"""
        metrics = self.model.val()
        print(f"mAP50-95: {metrics.box.map:.4f}")
        return metrics

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
    CONFIG_PATH = os.path.abspath("./bdd100k_yolo_640/bdd100k.yaml")
    PRETRAINED_MODEL = "yolov8n.pt"
    

    # Initialize
    finetuner = YOLOv8FineTuner(CONFIG_PATH)
    finetuner.prepare_model(PRETRAINED_MODEL, freeze_backbone=False)
    
    # Training parameters
    train_args = {
        'epochs': 10,
        'imgsz': 640,
        'batch': 16, # Max your GPU can handle (RTX 4070 can do ~batch=32)
        'lr0': 0.01,
        'workers': 4 if finetuner.device == '0' else 0,  # Only use workers for GPU
        'cos_lr': True,
        'weight_decay': 0.0005,
        # 'cache': 'disk',
        'label_smoothing': 0.1,
        'amp': True if finetuner.device == '0' else False,  # Mixed precision for GPU only
        'augment': True,  # Crucial for small datasets
        'patience': 15,   # Early stopping
        'fliplr': 0.5,    # Horizontal flip
        'copy_paste': 0.5,  # Copy-paste augmentation
        'mosaic': 0.5,    # Mosaic augmentation
        'mixup': 0.1,     # Mixup augmentation
        'hsv_h': 0.015,   # HSV hue augmentation
    
    }
    
    # Start training
    print(f"\nStarting training on {'GPU' if finetuner.device == '0' else 'CPU'}")
    print(f"Final training parameters: {train_args}")
    finetuner.train(**train_args)
    
    # Save the trained model
    finetuner.save_model("trained_yolov8n.pt")
    
    # Validate and export
    print("\nValidation results:")
    finetuner.validate()

    
    # Visualize
    sample_image = os.path.join(finetuner.data_dir, "images/val/0000f77c-62c2a288.jpg")
    if os.path.exists(sample_image):
        print("\nSample inference:")
        finetuner.visualize_results(sample_image)

if __name__ == "__main__":
    main()