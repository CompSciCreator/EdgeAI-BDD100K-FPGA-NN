
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import yaml
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

# Base directory in container
base_dir = "/workspace/bdd100k_yolo_640"

# Load model
weights_path = os.path.join(base_dir, "best.pt")
try:
    model = YOLO(weights_path)
except FileNotFoundError:
    print(f"Error: Weights file {weights_path} not found.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
model.model.eval()

# Read YAML
yaml_path = os.path.join(base_dir, "bdd100k_optimized.yaml")
try:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
        class_names = config['names']
except FileNotFoundError:
    print(f"Error: YAML file {yaml_path} not found.")
    exit(1)

# Transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
])

# Test image
img_path = os.path.join(base_dir, "images/test/cabc30fc-e7726578.jpg")
try:
    img_test = Image.open(img_path)
    if img_test.mode != 'RGB':
        img_test = img_test.convert('RGB')
except FileNotFoundError:
    print(f"Error: Test image {img_path} not found.")
    exit(1)
img_test = transform(img_test).unsqueeze(0)

# Run inference
with torch.no_grad():
    output = model.model(img_test)

# Parse raw output
try:
    # YOLOv8 raw output is typically a tuple of tensors; we take the first element
    if isinstance(output, (list, tuple)):
        output = output[0]  # Shape: (batch_size, num_predictions, num_outputs)

    # Debug: Print the shape of the output tensor
    print("Output tensor shape:", output.shape)

    # Split the output into components
    num_classes = len(class_names)  # Should be 10 based on YAML
    # Assuming the first 4 elements are box coords, 5th is objectness, rest are class probs
    box_coords = output[..., :4]  # (batch_size, num_predictions, 4)
    objectness = output[..., 4:5]  # (batch_size, num_predictions, 1)
    class_probs = output[..., 5:]  # (batch_size, num_predictions, num_classes)

    # Debug: Check the shape of class_probs
    print("Shape of class_probs:", class_probs.shape)
    print("Expected number of classes:", num_classes)

    # Apply sigmoid to objectness and class probabilities
    objectness = torch.sigmoid(objectness)
    class_probs = torch.sigmoid(class_probs)

    # Combine objectness with class probabilities to get confidence scores
    confidences = objectness * class_probs

    # Get the maximum confidence and corresponding class for each prediction
    probabilities, class_ids = confidences.max(dim=-1)

    # Debug: Print raw class IDs
    print("Predicted class IDs (before filtering):", class_ids)

    # Filter predictions with low confidence (e.g., threshold > 0.5)
    mask = probabilities > 0.5
    probabilities = probabilities[mask]
    class_ids = class_ids[mask]

    # Debug: Print filtered class IDs
    print("Predicted class IDs (after confidence filter):", class_ids)

    # Filter out invalid class IDs
    valid_mask = class_ids < num_classes  # Only keep class_ids < 10
    probabilities = probabilities[valid_mask]
    class_ids = class_ids[valid_mask]

    # Debug: Print final class IDs
    print("Predicted class IDs (after validity filter):", class_ids)

    if len(probabilities) == 0:
        print("No predictions above confidence threshold or with valid class IDs.")
        top5_prob, top5_catid = torch.tensor([]), torch.tensor([])
    else:
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, min(5, len(probabilities)))
        top5_catid = class_ids[top5_indices]

except Exception as e:
    print(f"Error parsing output: {e}")
    exit(1)

print("Original model predictions:")
if len(top5_prob) == 0:
    print("No detections.")
else:
    for i in range(len(top5_prob)):
        class_id = int(top5_catid[i])
        prob = top5_prob[i].item()
        print(f'{class_names[class_id]}: {prob * 100:.2f}%')


# pause here
input("Press Enter to continue...")



#Calibration images
calib_image_path = os.path.join(base_dir, "images/train")
calib_images = []
image_files = [os.path.join(calib_image_path, f) for f in os.listdir(calib_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))][:50]
for img_file in image_files:
    try:
        img = Image.open(img_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = transform(img)
        calib_images.append(img)
        print(f"Loaded {img_file}")
    except Exception as e:
        print(f"Skipping {img_file}: {e}")

if not calib_images:
    print("Error: No calibration images loaded.")
    exit(1)

calib_images_batch = torch.stack(calib_images)



# Quantize
try:
    quantizer = torch_quantizer("calib", model.model, (calib_images_batch))
    quant_model = quantizer.quant_model
except Exception as e:
    print(f"Quantization failed: {e}")
    exit(1)

# Evaluate quantized model
device = torch.device("cpu")
quant_model.eval()
quant_model = quant_model.to(device)
with torch.no_grad():
    output = quant_model(img_test)

# Parse quantized output
try:
    if isinstance(output, (list, tuple)):
        output = output[0]
    if hasattr(output, 'boxes'):
        probabilities = torch.nn.functional.softmax(output.boxes.conf, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, min(5, probabilities.size(1)))
    else:
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, min(5, probabilities.size(1)))
except Exception as e:
    print(f"Error parsing quantized output: {e}")
    exit(1)

print("Quantized model predictions:")
for i in range(top5_prob.size(0)):
    print(f'{class_names[top5_catid[i].item()]}: {top5_prob[i].item() * 100:.2f}%')

# Export quantization config
quantizer.export_quant_config()



# Test mode quantization
quantizer = torch_quantizer("test", model.model, (torch.randn([1, 3, 640, 640])))
quant_model = quantizer.quant_model
quant_model.eval()
quant_model = quant_model.to(device)
with torch.no_grad():
    output = quant_model(img_test)

# Parse test mode output
try:
    if isinstance(output, (list, tuple)):
        output = output[0]
    if hasattr(output, 'boxes'):
        probabilities = torch.nn.functional.softmax(output.boxes.conf, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, min(5, probabilities.size(1)))
    else:
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, min(5, probabilities.size(1)))
except Exception as e:
    print(f"Error parsing test mode output: {e}")
    exit(1)

print("Deployment check predictions:")
for i in range(top5_prob.size(0)):
    print(f'{class_names[top5_catid[i].item()]}: {top5_prob[i].item() * 100:.2f}%')

# Export
try:
    quantizer.export_xmodel(deploy_check=True)
    quantizer.export_onnx_model()
except Exception as e:
    print(f"Export failed: {e}")
    exit(1)

print("Quantization and export completed. The .xmodel file is ready for Kria KV260.")
