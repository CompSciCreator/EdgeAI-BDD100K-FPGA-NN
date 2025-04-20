import os
label_dir = "C:/Users/User1/EdgeAI-BDD100K-FPGA-NN/bdd100k_yolo_640_100k_v5/labels/train"
for f in os.listdir(label_dir):
    with open(os.path.join(label_dir, f)) as file:
        for line in file:
            parts = line.strip().split()
            assert len(parts) == 5, f"Invalid format in {f}: {line}"
            class_id = int(parts[0])
            assert 0 <= class_id <= 9, f"Invalid class_id {class_id} in {f}"
            coords = [float(x) for x in parts[1:]]
            assert all(0 <= x <= 1 for x in coords), f"Invalid coords in {f}: {coords}"
print("Train labels verified.")