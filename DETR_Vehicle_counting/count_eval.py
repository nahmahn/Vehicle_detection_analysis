import json
import os
from tqdm import tqdm
import numpy as np

def evaluate_counts():
    with open("annotations/instances_val.json", "r") as f:
        gt_data = json.load(f)
    
    with open("predictions/detr_predictions.json", "r") as f:
        pred_data = json.load(f)
    
    gt_counts = {}
    for ann in gt_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in gt_counts:
            gt_counts[img_id] = 0
        gt_counts[img_id] += 1
    
    pred_counts = {}
    for pred in pred_data:
        img_id = pred["image_id"]
        if img_id not in pred_counts:
            pred_counts[img_id] = 0
        pred_counts[img_id] += 1
    
    total_gt = sum(gt_counts.values())
    total_pred = sum(pred_counts.values())
    
    errors = []
    for img_id in gt_counts.keys():
        gt_count = gt_counts[img_id]
        pred_count = pred_counts.get(img_id, 0)
        error = abs(gt_count - pred_count)
        errors.append(error)
    
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    
    print("\nCount Evaluation Results:")
    print(f"Total ground truth vehicles: {total_gt}")
    print(f"Total predicted vehicles: {total_pred}")
    print(f"Mean absolute error: {mean_error:.2f}")
    print(f"Median absolute error: {median_error:.2f}")
    print(f"Maximum absolute error: {max_error}")
    
    results = {
        "total_gt": total_gt,
        "total_pred": total_pred,
        "mean_error": float(mean_error),
        "median_error": float(median_error),
        "max_error": int(max_error),
        "per_image_errors": {
            str(img_id): {
                "gt_count": gt_counts[img_id],
                "pred_count": pred_counts.get(img_id, 0),
                "error": abs(gt_counts[img_id] - pred_counts.get(img_id, 0))
            }
            for img_id in gt_counts.keys()
        }
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/count_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nDetailed results saved to results/count_eval_results.json")

if __name__ == "__main__":
    evaluate_counts()
