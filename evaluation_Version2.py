import numpy as np
import torch
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def convert_predictions_to_coco(predictions, image_ids):
    """
    Convert model predictions to COCO format for evaluation
    """
    coco_results = []
    for pred, image_id in zip(predictions, image_ids):
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            coco_results.append({
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            })
    
    return coco_results

def evaluate_map(predictions, ground_truth_file, image_ids):
    """
    Evaluate Mean Average Precision (MaP) at IoU threshold 0.5
    """
    # Load ground truth COCO format
    coco_gt = COCO(ground_truth_file)
    
    # Convert predictions to COCO format
    coco_results = convert_predictions_to_coco(predictions, image_ids)
    
    # Save results to a temporary file
    with open('temp_results.json', 'w') as f:
        json.dump(coco_results, f)
    
    # Load results in COCO format
    coco_dt = coco_gt.loadRes('temp_results.json')
    
    # Create COCO evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Optional: Set specific parameters
    coco_eval.params.imgIds = image_ids
    coco_eval.params.iouThrs = [0.5]  # IoU threshold
    
    # Evaluate
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Return mAP @ 0.5 IoU
    return coco_eval.stats[1]  # AP at IoU=0.50