from __future__ import annotations

import argparse
import json
import random
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

try:
    from .check_data import resolve_raw_layout
    from .coco_validation import validate_coco_data
    from .config_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT, get_required_path, load_config
except ImportError:  # pragma: no cover
    from check_data import resolve_raw_layout
    from coco_validation import validate_coco_data
    from config_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT, get_required_path, load_config


def load_coco_dataset(json_path: Path) -> Dict[str, Any]:
    with Path(json_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_file_name(file_name: str) -> str:
    return Path(file_name.replace("\\", "/")).name


def split_images(
    images: List[Dict[str, Any]], train_ratio: float = 0.8, seed: int = 42,
    annotations: List[Dict[str, Any]] | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(images) < 2:
        raise ValueError("at least two images are required for train/valid splitting")
    ordered = sorted(images, key=lambda item: (normalize_file_name(str(item["file_name"])), str(item["id"])))
    random.Random(seed).shuffle(ordered)
    split_index = min(max(round(len(ordered) * train_ratio), 1), len(ordered) - 1)
    train, valid = ordered[:split_index], ordered[split_index:]
    if annotations:
        # Keep every class represented in training where the dataset permits it.
        by_image: dict[Any, set[Any]] = {}
        for annotation in annotations:
            by_image.setdefault(annotation["image_id"], set()).add(annotation["category_id"])
        train_categories = {cat for image in train for cat in by_image.get(image["id"], set())}
        for image in list(valid):
            missing = {cat for cat in by_image.get(image["id"], set()) if cat not in train_categories}
            if missing and len(train) < len(images) - 1:
                valid.remove(image)
                train.append(image)
                train_categories.update(missing)
    return train, valid


def build_coco_subset(
    selected_images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    split_name: str,
    image_root: Path,
    output_root: Path,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    image_id_map: Dict[Any, int] = {}
    output_images: list[Dict[str, Any]] = []
    mapping: list[Dict[str, Any]] = []
    for new_id, image in enumerate(selected_images, start=1):
        original_name = normalize_file_name(str(image["file_name"]))
        source = image_root / original_name
        output_name = f"{new_id:06d}{source.suffix.lower()}"
        shutil.copy2(source, split_dir / output_name)
        image_id_map[image["id"]] = new_id
        output_images.append(
            {"id": new_id, "file_name": output_name, "width": image["width"], "height": image["height"]}
        )
        mapping.append(
            {
                "split": split_name,
                "source_image_id": image["id"],
                "source_file_name": original_name,
                "output_image_id": new_id,
                "output_file_name": output_name,
            }
        )

    source_to_output_category = {
        category["id"]: index for index, category in enumerate(categories, start=1)
    }
    output_categories = [
        {**category, "id": source_to_output_category[category["id"]]}
        for category in categories
    ]
    output_annotations: list[Dict[str, Any]] = []
    selected_ids = set(image_id_map)
    for annotation in annotations:
        if annotation.get("image_id") not in selected_ids:
            continue
        bbox = [float(value) for value in annotation["bbox"]]
        output_annotation = {
            "id": len(output_annotations) + 1,
            "image_id": image_id_map[annotation["image_id"]],
            "category_id": source_to_output_category[annotation["category_id"]],
            "bbox": bbox,
            "area": float(annotation.get("area", bbox[2] * bbox[3])),
            "iscrowd": int(annotation.get("iscrowd", 0)),
        }
        if "segmentation" in annotation:
            output_annotation["segmentation"] = annotation["segmentation"]
        output_annotations.append(output_annotation)
    return {
        "images": output_images,
        "annotations": output_annotations,
        "categories": output_categories,
    }, mapping


def _distribution(data: Dict[str, Any]) -> Dict[str, int]:
    counts = Counter(annotation["category_id"] for annotation in data["annotations"])
    return {str(category["id"]): counts[category["id"]] for category in data["categories"]}


def prepare_dataset(
    raw_base: Path,
    output_root: Path,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    overwrite: bool = False,
    decode_images: bool = True,
) -> Dict[str, Any]:
    raw_base = Path(raw_base).resolve()
    output_root = Path(output_root).resolve()
    annotation_path, image_root = resolve_raw_layout(raw_base)
    data = load_coco_dataset(annotation_path)
    raw_validation = validate_coco_data(
        data, image_root, decode_images=decode_images, source=str(annotation_path)
    )
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Processed dataset already exists: {output_root}. Use --overwrite to replace it safely."
        )

    temporary_root = output_root.with_name(f".{output_root.name}.tmp-{uuid.uuid4().hex}")
    backup_root = output_root.with_name(f".{output_root.name}.backup-{uuid.uuid4().hex}")
    train_images, valid_images = split_images(data["images"], train_ratio, seed, data["annotations"])
    try:
        temporary_root.mkdir(parents=True)
        train_data, train_mapping = build_coco_subset(
            train_images, data["annotations"], data["categories"], "train", image_root, temporary_root
        )
        valid_data, valid_mapping = build_coco_subset(
            valid_images, data["annotations"], data["categories"], "valid", image_root, temporary_root
        )
        for split, split_data in (("train", train_data), ("valid", valid_data)):
            annotation_output = temporary_root / split / "_annotations.coco.json"
            annotation_output.write_text(
                json.dumps(split_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            validate_coco_data(
                split_data,
                temporary_root / split,
                decode_images=decode_images,
                source=str(annotation_output),
            )
        train_names = {item["source_file_name"] for item in train_mapping}
        valid_names = {item["source_file_name"] for item in valid_mapping}
        overlap = sorted(train_names & valid_names)
        if overlap:
            raise ValueError(f"train/valid source image overlap: {overlap[:8]}")

        manifest = {
            "source_annotation": str(annotation_path),
            "source_image_root": str(image_root),
            "seed": seed,
            "train_ratio": train_ratio,
            "categories": [
                {**category, "id": index}
                for index, category in enumerate(data["categories"], start=1)
            ],
            "splits": {"train": train_mapping, "valid": valid_mapping},
        }
        summary = {
            "source_json": str(annotation_path),
            "output_root": str(output_root),
            "seed": seed,
            "train_ratio": train_ratio,
            "train_images": len(train_data["images"]),
            "valid_images": len(valid_data["images"]),
            "train_annotations": len(train_data["annotations"]),
            "valid_annotations": len(valid_data["annotations"]),
            "categories": len(data["categories"]),
            "train_category_distribution": _distribution(train_data),
            "valid_category_distribution": _distribution(valid_data),
            "valid_categories_present": sorted(
                {annotation["category_id"] for annotation in valid_data["annotations"]}
            ),
            "raw_validation": raw_validation,
        }
        (temporary_root / "split_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (temporary_root / "split_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists():
            output_root.replace(backup_root)
        temporary_root.replace(output_root)
        if backup_root.exists():
            shutil.rmtree(backup_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        if backup_root.exists() and not output_root.exists():
            backup_root.replace(output_root)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build deterministic RF-DETR train/valid COCO splits.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-decode", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    raw_root = get_required_path(config, "raw_data", "root", PROJECT_ROOT)
    output_root = get_required_path(config, "processed_data", "root", PROJECT_ROOT)
    split_cfg = config.get("data_split", {})
    prepare_dataset(
        raw_root,
        output_root,
        train_ratio=float(split_cfg.get("train_ratio", 0.8)),
        seed=int(config.get("project", {}).get("seed", 42)),
        overwrite=args.overwrite,
        decode_images=not args.no_decode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())