

import os
from PIL import Image
import random

# ✅ Your actual image path
SOURCE_DIR = r"C:/Users/Kevin/CSI_4110/calib_images"
DEST_DIR = "calib_images"
TARGET_SIZE = (640, 640)
NUM_SAMPLES = 200

os.makedirs(DEST_DIR, exist_ok=True)

image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
selected_files = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))

for i, fname in enumerate(selected_files):
    try:
        img = Image.open(os.path.join(SOURCE_DIR, fname)).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img.save(os.path.join(DEST_DIR, fname))

        if i % 20 == 0:
            print(f"[{i}/{NUM_SAMPLES}] Saved: {fname}")
    except Exception as e:
        print(f"Error processing {fname}: {e}")

print(f"\n Done! {len(selected_files)} images saved to {DEST_DIR}")