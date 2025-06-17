import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from transformers import DetrImageProcessor, DetrForObjectDetection
from tqdm import tqdm
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from multiprocessing import freeze_support

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((640, 640)),
    v2.ToDtype(torch.float32, scale=True)
])

class CocoDetWrapper(CocoDetection):
    def __getitem__(self, idx):
        img, targets = super().__getitem__(idx)
        if len(targets) == 0:
            return self[(idx + 1) % len(self)]  # loop safely
        image_id = targets[0]["image_id"]
        annotations = {
            "image_id": image_id,
            "annotations": targets
        }
        return transform(img), annotations

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load processor and model
    print("Loading model from checkpoint...")
    processor = DetrImageProcessor.from_pretrained("checkpoints/detr_best")
    model = DetrForObjectDetection.from_pretrained("checkpoints/detr_best")
    print(f"Model loaded. Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    model.to(device)
    model.eval()

    print("Loading validation dataset...")
    val_dataset = CocoDetWrapper(
        root="valid/images",
        annFile="annotations/instances_val.json"
    )
    print(f"Dataset loaded. Number of images: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    predictions = []
    with torch.no_grad():
        for images, annotations in tqdm(val_loader, desc="Evaluating"):
            try:
                inputs = processor(images=images, return_tensors="pt", do_rescale=False)
                
                outputs = model(**inputs)
                
                target_sizes = torch.tensor([img.shape[-2:] for img in images])
                results = processor.post_process_object_detection(
                    outputs, 
                    target_sizes=target_sizes, 
                    threshold=0.3  
                )
                
                for image_id, result in zip([ann["image_id"] for ann in annotations], results):
                    boxes = result["boxes"].numpy()
                    scores = result["scores"].numpy()
                    labels = result["labels"].numpy()
                    
                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        
                        if width * height < 100:  
                            continue
                            
                        predictions.append({
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [float(x1), float(y1), float(width), float(height)],
                            "score": float(score)
                        })
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    print(f"Total predictions: {len(predictions)}")
    
    os.makedirs("predictions", exist_ok=True)
    with open("predictions/detr_predictions.json", "w") as f:
        json.dump(predictions, f)

    print(f"Saved predictions to predictions/detr_predictions.json")

    print("Calculating mAP...")
    coco_gt = COCO("annotations/instances_val.json")
    coco_dt = coco_gt.loadRes("predictions/detr_predictions.json")
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\nPer-category AP:")
    for i, cat_id in enumerate(coco_gt.getCatIds()):
        cat_name = coco_gt.loadCats(cat_id)[0]['name']
        ap = coco_eval.eval['precision'][0, :, i, 0, 2].mean()
        print(f"{cat_name}: {ap:.4f}")

    metrics = {
        "mAP": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
        "mAP_75": float(coco_eval.stats[2]),
        "mAP_small": float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large": float(coco_eval.stats[5]),
        "per_category_ap": {
            coco_gt.loadCats(cat_id)[0]['name']: float(coco_eval.eval['precision'][0, :, i, 0, 2].mean())
            for i, cat_id in enumerate(coco_gt.getCatIds())
        }
    }

    with open("predictions/detr_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nDetailed metrics saved to predictions/detr_metrics.json")

if __name__ == '__main__':
    freeze_support()  
    evaluate()
