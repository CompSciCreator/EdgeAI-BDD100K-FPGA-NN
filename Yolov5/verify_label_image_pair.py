
import os
img_dir = " C:/Users/User1/EdgeAI-BDD100K-FPGA-NN/bdd100k_yolo_640_100k_v5/images/train"
label_dir = " C:/Users/User1/EdgeAI-BDD100K-FPGA-NN/bdd100k_yolo_640_100k_v5/labels/train"
for img_f in os.listdir(img_dir):
    if not img_f.endswith((".jpg", ".png")):
        continue
    label_f = img_f.replace(".jpg", ".txt").replace(".png", ".txt")
    assert os.path.exists(os.path.join(label_dir, label_f)), f"Missing label for {img_f}"
print("Image-label pairing verified.")