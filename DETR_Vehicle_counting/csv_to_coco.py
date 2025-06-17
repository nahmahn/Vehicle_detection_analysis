import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def convert_csv_to_coco(csv_path, output_json):
    df = pd.read_csv(csv_path)

    df["width"] = df["width"].astype(int)
    df["height"] = df["height"].astype(int)

    info = {
        "year": 2024,
        "version": "1.0",
        "description": "Vehicle Dataset",
        "contributor": "",
        "url": "",
        "date_created": "2024-01-01"
    }

    licenses = [{"id": 1, "name": "Unknown", "url": ""}]
    categories = [{"id": 1, "name": "vehicle", "supercategory": "vehicle"}]

    images = []
    annotations = []
    image_id_map = {}
    ann_id = 1

    grouped = df.groupby('filename')

    for idx, (filename, group) in enumerate(grouped):
        width = int(group.iloc[0]["width"])
        height = int(group.iloc[0]["height"])
        image_id = idx + 1
        image_id_map[filename] = image_id

        images.append({
            "file_name": filename,
            "id": image_id,
            "width": width,
            "height": height
        })

        for _, row in group.iterrows():
            xmin, ymin = float(row["xmin"]), float(row["ymin"])
            xmax, ymax = float(row["xmax"]), float(row["ymax"])
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            bbox = [xmin, ymin, bbox_width, bbox_height]

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })
            ann_id += 1

    coco_format = {
        "info": info,
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)


if __name__ == "__main__":
    convert_csv_to_coco("train/labels/_annotations.csv", "annotations/instances_train.json")
    convert_csv_to_coco("valid/labels/_annotations.csv", "annotations/instances_val.json")
