import onnxruntime
import numpy as np
import cv2
from utils.general import non_max_suppression
import torch

# Load the ONNX model
session = onnxruntime.InferenceSession("best_simplified.onnx")
input_name = session.get_inputs()[0].name

# Prepare a dummy or real image (preprocessed as during training)
img = cv2.imread("C:/Users/User1/EdgeAI-BDD100K-FPGA-NN\YOLOv5/yolov5_hackster/test_images/00a2f5b6-d4217a96.jpg")  # Replace with a BDD100K validation image
img = cv2.resize(img, (640, 640))
img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
img = img.transpose(2, 0, 1)  # HWC to CHW
img = np.expand_dims(img, axis=0)  # Add batch dimension: [1, 3, 640, 640]

# Run inference
outputs = session.run(None, {input_name: img})
print([out.shape for out in outputs])

# Combine outputs
outputs = np.concatenate([out.reshape(1, -1, out.shape[-1]) for out in outputs], axis=1)  # [1, 25200, 15]
print("Raw outputs:", outputs[0, :5, :])  # Print first 5 anchors
pred = non_max_suppression(torch.tensor(outputs), conf_thres=0.25, iou_thres=0.45)
print(pred)  # List of detections: [batch_size, (x1, y1, x2, y2, conf, cls)]