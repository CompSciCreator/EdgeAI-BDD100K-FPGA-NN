import cv2
import numpy as np
import onnxruntime
import time
import os
from pathlib import Path

def preprocess_image(image, input_size=(640, 640)):
    """Preprocess input image for YOLOv5"""
    img = cv2.resize(image, input_size)
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def postprocess_output(outputs, conf_thres=0.25, iou_thres=0.45, input_size=640):
    """Post-process YOLOv5 ONNX outputs, handling multiple output tensors"""
    predictions = np.concatenate([out.reshape(1, -1, out.shape[-1]) for out in outputs], axis=1)  # [1, num_anchors, nc+5]
    
    boxes = predictions[:, :, :4]  # [batch, num_anchors, 4]
    scores = predictions[:, :, 4]  # [batch, num_anchors]
    class_scores = predictions[:, :, 5:]  # [batch, num_anchors, num_classes]
    
    scores, class_ids = np.max(class_scores, axis=2), np.argmax(class_scores, axis=2)
    
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        conf_thres,
        iou_thres
    )
    
    if len(indices) > 0:
        indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]
        
        boxes[:, [0, 2]] *= input_size  # x1, x2
        boxes[:, [1, 3]] *= input_size  # y1, y2
    
    return boxes, scores, class_ids

def evaluate_images(test_images_path, yolov5_onnx_path, input_size=640):
    """Evaluate performance of all test images using YOLOv5 ONNX model"""
    
    # Load ONNX model with CUDA support if available
    try:
        session = onnxruntime.InferenceSession(
            yolov5_onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None, None, None
    
    # Get input name for ONNX model
    input_name = session.get_inputs()[0].name
    
    # Create output directory for processed images
    output_dir = "processed_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .jpg images
    image_files = [f for f in Path(test_images_path).glob("*.jpg")]
    if not image_files:
        print(f"No .jpg images found in {test_images_path}")
        return None, None, None
    
    total_fps = []
    total_preprocess_time = []
    total_inference_time = []
    total_postprocess_time = []
    processed_images = 0
    
    for image_path in image_files:
        start_time = time.time()
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
            
        # Preprocessing
        pre_process_start = time.time()
        input_tensor = preprocess_image(image, (input_size, input_size))
        pre_process_end = time.time()
        
        # ONNX inference
        dpu_start_onnx = time.time()
        try:
            outputs_onnx = session.run(None, {input_name: input_tensor})
        except Exception as e:
            print(f"Error during ONNX inference for {image_path}: {e}")
            continue
        dpu_end_onnx = time.time()
        
        # Post-processing
        decode_start = time.time()
        bboxes_onnx, scores_onnx, class_ids_onnx = postprocess_output(outputs_onnx, input_size=input_size)
        decode_end = time.time()
        
        # Calculate performance metrics
        preprocess_time = pre_process_end - pre_process_start
        inference_time = dpu_end_onnx - dpu_start_onnx
        postprocess_time = decode_end - decode_start
        total_time = decode_end - start_time
        fps = 1 / total_time if total_time > 0 else 0
        
        # Store metrics
        total_fps.append(fps)
        total_preprocess_time.append(preprocess_time)
        total_inference_time.append(inference_time)
        total_postprocess_time.append(postprocess_time)
        processed_images += 1
        
        # Print results for each image
        print(f"\nResults for {image_path} (ONNX):")
        print("bboxes of detected objects: {}".format(bboxes_onnx))
        print("scores of detected objects: {}".format(scores_onnx))
        print("Details of detected objects: {}".format(class_ids_onnx))
        print("Pre-processing time: {:.4f} seconds".format(preprocess_time))
        print("Inference time: {:.4f} seconds".format(inference_time))
        print("Post-process time: {:.4f} seconds".format(postprocess_time))
        print("Total run time: {:.4f} seconds".format(total_time))
        print("Performance: {:.4f} FPS".format(fps))
        print(" ")
        
        # Save results with bounding boxes
        output_path = os.path.join(output_dir, f"processed_{image_path.name}")
        output_image = image.copy()
        for bbox in bboxes_onnx:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(output_path, output_image)
    
    # Print summary and average metrics
    print(f"\nProcessed {processed_images} out of {len(image_files)} images")
    if total_fps:
        print("\nAverage Performance Metrics (ONNX):")
        print(f"Average FPS: {np.mean(total_fps):.4f}")
        print(f"Average Pre-processing time: {np.mean(total_preprocess_time):.4f} seconds")
        print(f"Average Inference time: {np.mean(total_inference_time):.4f} seconds")
        print(f"Average Post-process time: {np.mean(total_postprocess_time):.4f} seconds")
    else:
        print("No images were successfully processed")
    
    return bboxes_onnx, scores_onnx, class_ids_onnx

if __name__ == "__main__":
    test_images_path = "test_images"  # Directory containing test images
    yolov5_onnx_path = "best_model_no_int8.onnx"
    
    bboxes_onnx, scores_onnx, class_ids_onnx = evaluate_images(
        test_images_path,
        yolov5_onnx_path
    )