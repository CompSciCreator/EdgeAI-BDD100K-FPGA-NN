from ultralytics import YOLO

model = YOLO('trained_yolov8_200.pt')
model.export(format='onnx', opset=11, simplify=True, dynamic=False)
