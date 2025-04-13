import os
import numpy as np
import cv2

def calib_input(iter=100):
    image_dir = "./calib_images"
    image_list = sorted(os.listdir(image_dir))
    print(f"[INFO] Found {len(image_list)} images in {image_dir}")
    print(f"[INFO] Preparing {iter} calibration samples...")

    samples = []
    for i in range(iter):
        img_path = os.path.join(image_dir, image_list[i % len(image_list)])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        samples.append({'images': img})
        print(f"[DEBUG] Sample {i+1}: {img_path} -> shape {img.shape}")

    print("[INFO] Calibration data ready.\n")
    return samples 
