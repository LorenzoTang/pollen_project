from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _decode_image(path: Path) -> tuple[int, int]:
    """Decode an image and return its dimensions."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for image integrity checks. Install project dependencies first."
        ) from exc
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except Exception as exc:
        raise ValueError(f"Image cannot be decoded: {path}: {exc}") from exc


def validate_coco_data(
    data: Dict[str, Any],
    image_root: Path,
    *,
    decode_images: bool = True,
    source: str = "COCO data",
) -> Dict[str, Any]:
    """Validate IDs, boxes, category references, files, and decoded dimensions."""
    image_root = Path(image_root)
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        raise ValueError(f"{source} root must be a JSON object")
    images = data.get("images")
    annotations = data.get("annotations")
    categories = data.get("categories")
    if not isinstance(images, list) or not images:
        errors.append("images must be a non-empty list")
        images = []
    if not isinstance(annotations, list) or not annotations:
        errors.append("annotations must be a non-empty list")
        annotations = []
    if not isinstance(categories, list) or not categories:
        errors.append("categories must be a non-empty list")
        categories = []

    category_ids: set[Any] = set()
    category_names: dict[Any, str] = {}
    for index, category in enumerate(categories):
        category_id = category.get("id") if isinstance(category, dict) else None
        name = category.get("name") if isinstance(category, dict) else None
        if category_id is None or category_id in category_ids:
            errors.append(f"category[{index}] has a missing or duplicate id: {category_id!r}")
        else:
            category_ids.add(category_id)
        if not isinstance(name, str) or not name.strip():
            errors.append(f"category[{index}] has an invalid name")
        else:
            category_names[category_id] = name

    image_ids: set[Any] = set()
    file_names: set[str] = set()
    dimensions: dict[Any, tuple[float, float]] = {}
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            errors.append(f"image[{index}] must be an object")
            continue
        image_id = image.get("id")
        file_name = image.get("file_name")
        width, height = image.get("width"), image.get("height")
        if image_id is None or image_id in image_ids:
            errors.append(f"image[{index}] has a missing or duplicate id: {image_id!r}")
        else:
            image_ids.add(image_id)
        if not isinstance(file_name, str) or not file_name.strip():
            errors.append(f"image[{index}] has an invalid file_name")
            continue
        normalized_name = Path(file_name.replace("\\", "/")).name
        if normalized_name in file_names:
            errors.append(f"duplicate image basename: {normalized_name}")
        file_names.add(normalized_name)
        if not _is_number(width) or not _is_number(height) or width <= 0 or height <= 0:
            errors.append(f"image id={image_id!r} has invalid dimensions: {width}x{height}")
            continue
        dimensions[image_id] = (float(width), float(height))
        image_path = image_root / normalized_name
        if not image_path.is_file():
            errors.append(f"missing image for id={image_id!r}: {image_path}")
            continue
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            warnings.append(f"unusual image extension: {image_path.name}")
        if decode_images:
            decoded_size = _decode_image(image_path)
            if decoded_size != (int(width), int(height)):
                errors.append(
                    f"dimension mismatch for {normalized_name}: COCO={int(width)}x{int(height)}, "
                    f"decoded={decoded_size[0]}x{decoded_size[1]}"
                )

    annotation_ids: set[Any] = set()
    category_distribution: Counter[Any] = Counter()
    images_with_annotations: set[Any] = set()
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            errors.append(f"annotation[{index}] must be an object")
            continue
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")
        if annotation_id is None or annotation_id in annotation_ids:
            errors.append(f"annotation[{index}] has a missing or duplicate id: {annotation_id!r}")
        else:
            annotation_ids.add(annotation_id)
        if image_id not in image_ids:
            errors.append(f"annotation id={annotation_id!r} references unknown image_id={image_id!r}")
        else:
            images_with_annotations.add(image_id)
        if category_id not in category_ids:
            errors.append(f"annotation id={annotation_id!r} references unknown category_id={category_id!r}")
        else:
            category_distribution[category_id] += 1
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_number(v) for v in bbox):
            errors.append(f"annotation id={annotation_id!r} has invalid bbox: {bbox!r}")
            continue
        x, y, width, height = (float(value) for value in bbox)
        if width <= 0 or height <= 0:
            errors.append(f"annotation id={annotation_id!r} has non-positive bbox: {bbox!r}")
        image_size = dimensions.get(image_id)
        if image_size and (
            x < 0 or y < 0 or x + width > image_size[0] + 1e-6 or y + height > image_size[1] + 1e-6
        ):
            errors.append(f"annotation id={annotation_id!r} bbox is outside image bounds: {bbox!r}")

    unannotated = image_ids - images_with_annotations
    if unannotated:
        warnings.append(f"{len(unannotated)} images have no annotations")
    empty_categories = sorted(category_ids - set(category_distribution), key=str)
    if empty_categories:
        warnings.append(f"categories without annotations: {empty_categories}")
    report = {
        "status": "passed" if not errors else "failed",
        "source": source,
        "image_root": str(image_root.resolve()),
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "category_distribution": {
            str(category_id): {
                "name": category_names.get(category_id, ""),
                "annotations": category_distribution.get(category_id, 0),
            }
            for category_id in sorted(category_ids, key=str)
        },
        "warnings": list(dict.fromkeys(warnings)),
        "errors": errors,
    }
    if errors:
        preview = "; ".join(errors[:8])
        suffix = f" (+{len(errors) - 8} more)" if len(errors) > 8 else ""
        raise ValueError(f"{source} validation failed: {preview}{suffix}")
    return report


def validate_coco_file(
    annotation_path: Path,
    image_root: Path | None = None,
    *,
    decode_images: bool = True,
) -> Dict[str, Any]:
    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotation file does not exist: {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return validate_coco_data(
        data,
        image_root or annotation_path.parent,
        decode_images=decode_images,
        source=str(annotation_path.resolve()),
    )