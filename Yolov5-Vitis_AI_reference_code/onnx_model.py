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
    
    # Extract boxes, scores, and class scores
    boxes = predictions[:, :, :4]  # [batch, num_anchors, 4] (x_center, y_center, width, height)
    scores = predictions[:, :, 4]  # [batch, num_anchors]
    class_scores = predictions[:, :, 5:]  # [batch, num_anchors, num_classes]
    
    # Get max class scores and class IDs
    scores, class_ids = np.max(class_scores, axis=2), np.argmax(class_scores, axis=2)
    
    # Filter by confidence
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) > 0:
        # Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = x_center - width/2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = y_center - height/2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = x_center + width/2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = y_center + height/2
        
        # Clamp boxes to image boundaries [0, input_size]
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, input_size)  # x1, x2
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, input_size)  # y1, y2
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            conf_thres,
            iou_thres
        )
        
        if len(indices) > 0:
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            boxes_xyxy = boxes_xyxy[indices]
            scores = scores[indices]
            class_ids = class_ids[indices]
        else:
            boxes_xyxy, scores, class_ids = np.array([]), np.array([]), np.array([])
    else:
        boxes_xyxy, scores, class_ids = np.array([]), np.array([]), np.array([])
    
    return boxes_xyxy, scores, class_ids

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def compute_ap(recall, precision):
    """Compute Average Precision from recall and precision arrays"""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    
    i = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    return ap

def evaluate_images(test_images_path, yolov5_onnx_path, annotation_dir, input_size=640, nc=10):
    """Evaluate performance and mAP of all test images using YOLOv5 ONNX model on CPU"""
    
    # Load ONNX model
    try:
        session = onnxruntime.InferenceSession(
            yolov5_onnx_path,
            providers=['CPUExecutionProvider']
        )
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None, None, None
    
    input_name = session.get_inputs()[0].name
    
    # Create output directory
    output_dir = "processed_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_files = [f for f in Path(test_images_path).glob("*.jpg")]
    if not image_files:
        print(f"No .jpg images found in {test_images_path}")
        return None, None, None
    
    total_fps = []
    total_preprocess_time = []
    total_inference_time = []
    total_postprocess_time = []
    processed_images = 0
    all_predictions = []
    all_ground_truths = []
    
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
        
        # Inference
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
        
        # Performance metrics
        preprocess_time = pre_process_end - pre_process_start
        inference_time = dpu_end_onnx - dpu_start_onnx
        postprocess_time = decode_end - decode_start
        total_time = decode_end - start_time
        fps = 1 / total_time if total_time > 0 else 0
        
        total_fps.append(fps)
        total_preprocess_time.append(preprocess_time)
        total_inference_time.append(inference_time)
        total_postprocess_time.append(postprocess_time)
        processed_images += 1
        
        # Load ground truth annotations
        annotation_path = Path(annotation_dir) / f"{image_path.stem}.txt"
        gt_boxes = []
        gt_classes = []
        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                for line in f:
                    try:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        # Convert YOLO format to [x1, y1, x2, y2]
                        x1 = (x_center - width / 2) * input_size
                        y1 = (y_center - height / 2) * input_size
                        x2 = (x_center + width / 2) * input_size
                        y2 = (y_center + height / 2) * input_size
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_classes.append(int(class_id))
                    except ValueError:
                        print(f"Invalid annotation format in {annotation_path}")
                        continue
        else:
            print(f"No annotation file found for {image_path}")
        
        # Store predictions and ground truths
        pred_boxes = bboxes_onnx.tolist() if bboxes_onnx.size > 0 else []
        pred_scores = scores_onnx.tolist() if scores_onnx.size > 0 else []
        pred_classes = class_ids_onnx.tolist() if class_ids_onnx.size > 0 else []
        all_predictions.append({'boxes': pred_boxes, 'scores': pred_scores, 'classes': pred_classes})
        all_ground_truths.append({'boxes': gt_boxes, 'classes': gt_classes})
        
        # Print results
        print(f"\nResults for {image_path} (ONNX, CPU):")
        print("bboxes of detected objects: {}".format(bboxes_onnx if bboxes_onnx.size > 0 else "[]"))
        print("scores of detected objects: {}".format(scores_onnx if scores_onnx.size > 0 else "[]"))
        print("Details of detected objects: {}".format(class_ids_onnx if class_ids_onnx.size > 0 else "[]"))
        print("Pre-processing time: {:.4f} seconds".format(preprocess_time))
        print("Inference time: {:.4f} seconds".format(inference_time))
        print("Post-process time: {:.4f} seconds".format(postprocess_time))
        print("Total run time: {:.4f} seconds".format(total_time))
        print("Performance: {:.4f} FPS".format(fps))
        print(" ")
        
        # Save processed image
        output_path = os.path.join(output_dir, f"processed_{image_path.name}")
        output_image = image.copy()
        for bbox in bboxes_onnx:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(output_path, output_image)
    
    # Compute mAP@0.5 and mAP@0.5:0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # For mAP@0.5:0.95
    mAPs = []
    mAP_50 = 0.0
    
    for iou_thres in iou_thresholds:
        aps = []
        for class_id in range(nc):
            true_positives = []
            scores = []
            num_gt = 0
            
            for pred, gt in zip(all_predictions, all_ground_truths):
                pred_boxes = np.array(pred['boxes'])
                pred_scores = np.array(pred['scores'])
                pred_classes = np.array(pred['classes'])
                gt_boxes = np.array(gt['boxes'])
                gt_classes = np.array(gt['classes'])
                
                # Filter for current class
                pred_mask = pred_classes == class_id
                pred_boxes = pred_boxes[pred_mask]
                pred_scores = pred_scores[pred_mask]
                
                gt_mask = gt_classes == class_id
                gt_boxes = gt_boxes[gt_mask]
                num_gt += len(gt_boxes)
                
                # Sort predictions by score
                if len(pred_scores) > 0:
                    indices = np.argsort(-pred_scores)
                    pred_boxes = pred_boxes[indices]
                    pred_scores = pred_scores[indices]
                    
                    matched = set()
                    for i, pred_box in enumerate(pred_boxes):
                        best_iou = 0
                        best_gt_idx = -1
                        for j, gt_box in enumerate(gt_boxes):
                            if j in matched:
                                continue
                            iou = compute_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j
                        if best_iou >= iou_thres:
                            true_positives.append(1)
                            matched.add(best_gt_idx)
                        else:
                            true_positives.append(0)
                        scores.append(pred_scores[i])
            
            if num_gt == 0:
                continue
                
            true_positives = np.array(true_positives)
            scores = np.array(scores)
            if len(scores) == 0:
                aps.append(0.0)
                continue
                
            indices = np.argsort(-scores)
            true_positives = true_positives[indices]
            
            false_positives = 1 - true_positives
            cum_tp = np.cumsum(true_positives)
            cum_fp = np.cumsum(false_positives)
            
            recall = cum_tp / (num_gt + 1e-6)
            precision = cum_tp / (cum_tp + cum_fp + 1e-6)
            
            ap = compute_ap(recall, precision)
            aps.append(ap)
        
        if aps:
            mAP = np.mean(aps)
            mAPs.append(mAP)
            if iou_thres == 0.5:
                mAP_50 = mAP
    
    mAP_50_95 = np.mean(mAPs) if mAPs else 0.0
    
    # Print results
    print(f"\nProcessed {processed_images} out of {len(image_files)} images")
    if total_fps:
        print("\nAverage Performance Metrics (ONNX, CPU):")
        print(f"Average FPS: {np.mean(total_fps):.4f}")
        print(f"Average Pre-processing time: {np.mean(total_preprocess_time):.4f} seconds")
        print(f"Average Inference time: {np.mean(total_inference_time):.4f} seconds")
        print(f"Average Post-process time: {np.mean(total_postprocess_time):.4f} seconds")
    else:
        print("No images were successfully processed")
    
    print("\nmAP Metrics (ONNX, CPU):")
    print(f"mAP@0.5: {mAP_50:.4f}")
    print(f"mAP@0.5:0.95: {mAP_50_95:.4f}")
    
    return bboxes_onnx, scores_onnx, class_ids_onnx

if __name__ == "__main__":
    test_images_path = "test_images"
    yolov5_onnx_path = "best_model_no_int8.onnx"
    annotation_dir = "test_labels"  # Directory with .txt annotation files
    
    bboxes_onnx, scores_onnx, class_ids_onnx = evaluate_images(
        test_images_path,
        yolov5_onnx_path,
        annotation_dir
    )