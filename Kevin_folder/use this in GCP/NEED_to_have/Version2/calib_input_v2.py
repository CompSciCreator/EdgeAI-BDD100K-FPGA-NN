import os
import numpy as np
import cv2

def calib_input_v2(iter=0):
    image_dir = "./calib_images"
    image_list = sorted(os.listdir(image_dir))
    if not image_list:
        dummy = np.zeros((1,3,640,640), dtype=np.float32)
        return {"images": dummy}

    idx = (iter - 1) % len(image_list) if iter > 0 else 0
    img_path = os.path.join(image_dir, image_list[idx])
    print("[INFO] Returning sample {} from {}".format(iter, img_path))
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640,640))
    img = img.astype(np.float32) / 255.0

    # Convert to channels first: (640,640,3) -> (3,640,640)
    img = np.transpose(img, (2,0,1))

    # Expand to (1,3,640,640)
    img = np.expand_dims(img, axis=0)
    return {"images": img}
