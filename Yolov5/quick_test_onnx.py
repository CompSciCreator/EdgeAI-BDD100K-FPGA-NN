import onnxruntime
import numpy as np

session = onnxruntime.InferenceSession("best.onnx")
input_name = session.get_inputs()[0].name
img = np.zeros((1, 3, 640, 640), dtype=np.float32)
outputs = session.run(None, {input_name: img})[0]
print(outputs.shape)