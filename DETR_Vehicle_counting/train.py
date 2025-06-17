import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import DetrConfig
from transformers.models.detr.modeling_detr import build_detr, build_position_encoding, DetrForObjectDetection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print("Loading EfficientNet-B3 backbone...")
effnet = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
effnet.requires_grad_(True)
features = nn.Sequential(*list(effnet.features))


class EfficientNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = features
        self.projection = nn.Conv2d(384, 256, kernel_size=1)  # 384 = last channel in B3

    def forward(self, x):
        x = self.body(x)
        x = self.projection(x)
        return {"0": x}


print("Building DETR with EfficientNet backbone...")
config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
config.num_labels = 2
position_encoding = build_position_encoding(config)
detr_core = build_detr(config)

class DetrWithEfficientNet(DetrForObjectDetection):
    def __init__(self, config, backbone):
        super().__init__(config)
        self.model.backbone.body = backbone


backbone = EfficientNetBackbone()
model = DetrWithEfficientNet(config, backbone)
model.to(device)


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

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = CocoDetWrapper(
    root="train/images",
    annFile="annotations/instances_train.json"
)
val_dataset = CocoDetWrapper(
    root="valid/images",
    annFile="annotations/instances_val.json"
)

train_loader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

num_epochs = 10
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

    for images, annotations in train_pbar:
        try:
            
            inputs = {
                "pixel_values": torch.stack(images).to(device),
                "labels": [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in a.items()} for a in annotations]
            }

            outputs = model(**inputs)
            loss = sum(outputs.losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        except Exception as e:
            print(f"Training error: {e}")
            continue

    avg_train_loss = total_train_loss / len(train_loader)


    model.eval()
    total_val_loss = 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

    with torch.no_grad():
        for images, annotations in val_pbar:
            try:
                inputs = {
                    "pixel_values": torch.stack(images).to(device),
                    "labels": [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in a.items()} for a in annotations]
                }
                outputs = model(**inputs)
                loss = sum(outputs.losses.values())
                total_val_loss += loss.item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            except Exception as e:
                print(f"Validation error: {e}")
                continue

    avg_val_loss = total_val_loss / len(val_loader)

    scheduler.step()

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")


    save_path = f"checkpoints/detr_efficientnet_epoch_{epoch+1}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_save_path = "checkpoints/detr_efficientnet_best"
        os.makedirs(best_save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(best_save_path, "pytorch_model.bin"))
        print(f"Best model saved with val loss: {best_val_loss:.4f}")

print("Training complete.")
