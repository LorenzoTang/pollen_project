from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


RAW_BASE = Path(r"data/raw/标签数据(1)")
COCO_JSON = RAW_BASE / "coco" / "dataset.json"
IMAGE_ROOT = RAW_BASE / "label"
OUTPUT_ROOT = Path(r"data/processed/rfdetr_pollen")

TRAIN_RATIO = 0.8
SEED = 42


def load_coco_dataset(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_file_name(file_name: str) -> str:
    return Path(file_name).name


def find_image_path(file_name: str) -> Path:
    return IMAGE_ROOT / normalize_file_name(file_name)


def validate_dataset(data: Dict[str, Any]) -> None:
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    category_ids = {cat.get("id") for cat in categories if "id" in cat}
    image_ids = {img.get("id") for img in images if "id" in img}

    if not images:
        raise ValueError("dataset.json does not contain images")
    if not annotations:
        raise ValueError("dataset.json does not contain annotations")
    if not categories:
        raise ValueError("dataset.json does not contain categories")

    for img in images:
        file_name = img.get("file_name")
        if not file_name:
            raise ValueError(f"image entry missing file_name: {img}")
        image_path = find_image_path(file_name)
        if not image_path.exists():
            raise FileNotFoundError(f"missing image file: {image_path}")

    for ann in annotations:
        if ann.get("image_id") not in image_ids:
            raise ValueError(f"annotation references unknown image_id: {ann}")
        if "bbox" not in ann or ann.get("bbox") is None:
            raise ValueError(f"annotation missing bbox: {ann}")
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"invalid bbox format: {ann}")
        if ann.get("category_id") not in category_ids:
            raise ValueError(f"annotation references unknown category_id: {ann}")


def split_images(images: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(SEED)
    shuffled = images[:]
    rng.shuffle(shuffled)
    split_index = int(len(shuffled) * TRAIN_RATIO)
    return shuffled[:split_index], shuffled[split_index:]


def build_coco_subset(
    split_images_list: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    split_name: str,
) -> Dict[str, Any]:
    image_id_map: Dict[Any, int] = {}
    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []

    split_image_dir = OUTPUT_ROOT / split_name / "images"
    split_image_dir.mkdir(parents=True, exist_ok=True)

    for new_id, img in enumerate(split_images_list, start=1):
        src = find_image_path(img["file_name"])
        dst = split_image_dir / normalize_file_name(img["file_name"])
        shutil.copy2(src, dst)

        image_id_map[img["id"]] = new_id
        coco_images.append(
            {
                "id": new_id,
                "file_name": dst.name,
                "width": img.get("width"),
                "height": img.get("height"),
            }
        )

    ann_id = 1
    split_image_ids = set(image_id_map.keys())
    for ann in annotations:
        if ann.get("image_id") not in split_image_ids:
            continue

        bbox = ann.get("bbox")
        coco_ann = {
            "id": ann_id,
            "image_id": image_id_map[ann["image_id"]],
            "category_id": ann["category_id"],
            "bbox": bbox,
            "iscrowd": ann.get("iscrowd", 0),
            "area": ann.get("area", bbox[2] * bbox[3]),
        }

        if "segmentation" in ann:
            coco_ann["segmentation"] = ann["segmentation"]

        coco_annotations.append(coco_ann)
        ann_id += 1

    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }


def prepare_dataset() -> None:
    data = load_coco_dataset(COCO_JSON)
    validate_dataset(data)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "annotations").mkdir(parents=True, exist_ok=True)

    train_images, val_images = split_images(images)

    train_coco = build_coco_subset(train_images, annotations, categories, "train")
    val_coco = build_coco_subset(val_images, annotations, categories, "val")

    with (OUTPUT_ROOT / "annotations" / "train.json").open("w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)

    with (OUTPUT_ROOT / "annotations" / "val.json").open("w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False, indent=2)

    summary = {
        "source_json": str(COCO_JSON),
        "output_root": str(OUTPUT_ROOT),
        "seed": SEED,
        "train_images": len(train_coco["images"]),
        "val_images": len(val_coco["images"]),
        "train_annotations": len(train_coco["annotations"]),
        "val_annotations": len(val_coco["annotations"]),
        "categories": len(categories),
    }

    with (OUTPUT_ROOT / "annotations" / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare_dataset()