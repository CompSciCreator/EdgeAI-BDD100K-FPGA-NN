from PIL import Image
import os
img_dir = "C:/Users/User1/EdgeAI-BDD100K-FPGA-NN/bdd100k_yolo_640_100k_v5/images/train"
for f in os.listdir(img_dir):
    img = Image.open(os.path.join(img_dir, f))
    assert img.size == (640, 640), f"Invalid size in {f}: {img.size}"
print("Train images verified.")