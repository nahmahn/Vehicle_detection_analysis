import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from transformers import DetrImageProcessor, DetrForObjectDetection
from tqdm import tqdm
import json
import os
import cv2
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = DetrImageProcessor.from_pretrained("checkpoints/detr_best")
model = DetrForObjectDetection.from_pretrained("checkpoints/detr_best")
model.to(device)
model.eval()

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((640, 640)),
    v2.ToDtype(torch.float32, scale=True)
])

class CocoDetWrapper(CocoDetection):
    def __getitem__(self, idx):
        img, targets = super().__getitem__(idx)
        if len(targets) == 0:
            return self[(idx + 1) % len(self)]
        image_id = targets[0]["image_id"]
        annotations = {
            "image_id": image_id,
            "annotations": targets
        }
        return transform(img), annotations

val_dataset = CocoDetWrapper(
    root="valid/images",
    annFile="annotations/instances_val.json"
)

with open("predictions/detr_predictions.json", "r") as f:
    predictions = json.load(f)

os.makedirs("vis_preds", exist_ok=True)

with torch.no_grad():
    for idx in tqdm(range(len(val_dataset)), desc="Visualizing"):
        img, annotations = val_dataset[idx]
        image_id = annotations["image_id"]
        
        img_preds = [p for p in predictions if p["image_id"] == image_id]
        
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        for ann in annotations["annotations"]:
            bbox = ann["bbox"]
            x, y, w, h = [int(coord) for coord in bbox]
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        for pred in img_preds:
            bbox = pred["bbox"]
            score = pred["score"]
            x, y, w, h = [int(coord) for coord in bbox]
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img_np, f"{score:.2f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imwrite(f"vis_preds1/{image_id}.jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

print("Visualizations saved to vis_preds/")
