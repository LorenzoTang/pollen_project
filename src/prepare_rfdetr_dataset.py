from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_RAW_BASE = Path(r"data/raw/标签数据(1)")
DEFAULT_OUTPUT_ROOT = Path(r"data/processed/rfdetr_pollen")

RAW_BASE = DEFAULT_RAW_BASE
COCO_JSON = RAW_BASE / "coco" / "dataset.json"
IMAGE_ROOT = RAW_BASE / "label"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT

TRAIN_RATIO = 0.8
SEED = 42


def load_coco_dataset(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_file_name(file_name: str) -> str:
    return Path(file_name).name


def find_image_path(file_name: str, image_root: Path = IMAGE_ROOT) -> Path:
    return image_root / normalize_file_name(file_name)


def validate_dataset(data: Dict[str, Any], image_root: Path = IMAGE_ROOT) -> None:
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
        image_path = find_image_path(file_name, image_root=image_root)
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


def standardize_categories(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            **cat,
            "name": cat.get("name"),
        }
        for cat in categories
    ]


def build_coco_subset(
    split_images_list: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    split_name: str,
    image_root: Path = IMAGE_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> Dict[str, Any]:
    image_id_map: Dict[Any, int] = {}
    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []

    split_image_dir = output_root / split_name
    split_image_dir.mkdir(parents=True, exist_ok=True)

    for new_id, img in enumerate(split_images_list, start=1):
        src = find_image_path(img["file_name"], image_root=image_root)
        dst = split_image_dir / f"{new_id:06d}{src.suffix.lower()}"
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
        "categories": standardize_categories(categories),
    }


def write_coco_json(split_name: str, coco_data: Dict[str, Any], output_root: Path = OUTPUT_ROOT) -> None:
    annotations_path = output_root / split_name / "_annotations.coco.json"
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with annotations_path.open("w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)


def prepare_dataset(raw_base: Path | None = None, output_root: Path | None = None) -> Dict[str, Any]:
    raw_base = Path(raw_base) if raw_base is not None else DEFAULT_RAW_BASE
    output_root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT

    coco_json = raw_base / "coco" / "dataset.json"
    image_root = raw_base / "label"

    data = load_coco_dataset(coco_json)
    validate_dataset(data, image_root=image_root)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_images, val_images = split_images(images)

    train_coco = build_coco_subset(train_images, annotations, categories, "train", image_root=image_root, output_root=output_root)
    val_coco = build_coco_subset(val_images, annotations, categories, "valid", image_root=image_root, output_root=output_root)

    write_coco_json("train", train_coco, output_root=output_root)
    write_coco_json("valid", val_coco, output_root=output_root)

    summary = {
        "source_json": str(coco_json),
        "output_root": str(output_root),
        "seed": SEED,
        "train_images": len(train_coco["images"]),
        "val_images": len(val_coco["images"]),
        "train_annotations": len(train_coco["annotations"]),
        "val_annotations": len(val_coco["annotations"]),
        "categories": len(categories),
    }

    summary_path = output_root / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    prepare_dataset()