import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("trained_yolov8_200.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("yolov8n_tf")
