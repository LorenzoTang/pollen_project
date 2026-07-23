from pathlib import Path
import json
import sys
from typing import Any, Dict

import yaml
from PIL import Image
from rfdetr import RFDETRMedium, RFDETRNano
from pycocotools.coco import COCO


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "rfdetr_baseline.yaml"


SUPPORTED_MODEL_TYPES = {"nano", "medium"}


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "nano":
        return RFDETRNano()
    if model_type == "medium":
        return RFDETRMedium()
    raise ValueError(f"Unsupported model type: {model_type}")


def find_first_val_image(dataset_path: Path) -> Path:
    val_dir = dataset_path / "val" / "images"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation image directory does not exist: {val_dir}")

    candidates = sorted(
        p for p in val_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    )
    if not candidates:
        raise FileNotFoundError(f"No validation images found in: {val_dir}")
    return candidates[0]


def load_validation_images(dataset_path: Path):
    ann_path = dataset_path / "valid" / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Validation annotation file does not exist: {ann_path}")

    coco = COCO(str(ann_path))
    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)
    return coco, images


def summarize_detections(detections: Any) -> Dict[str, Any]:
    summary = {
        "type": type(detections).__name__,
        "num_detections": len(detections),
        "has_xyxy": hasattr(detections, "xyxy"),
        "has_confidence": hasattr(detections, "confidence"),
        "has_class_id": hasattr(detections, "class_id"),
    }

    if hasattr(detections, "xyxy"):
        summary["xyxy"] = detections.xyxy.tolist() if getattr(detections, "xyxy") is not None else None
    if hasattr(detections, "confidence"):
        conf = getattr(detections, "confidence")
        summary["confidence"] = conf.tolist() if conf is not None else None
    if hasattr(detections, "class_id"):
        class_id = getattr(detections, "class_id")
        summary["class_id"] = class_id.tolist() if class_id is not None else None

    return summary


def detections_to_raw_list(detections: Any):
    raw = []
    xyxy = getattr(detections, "xyxy", None)
    confidence = getattr(detections, "confidence", None)
    class_id = getattr(detections, "class_id", None)

    if xyxy is None:
        return raw

    for idx, bbox in enumerate(xyxy):
        item = {"bbox_xyxy": [float(v) for v in bbox.tolist()]}
        if confidence is not None and idx < len(confidence):
            item["confidence"] = float(confidence[idx])
        else:
            item["confidence"] = None
        if class_id is not None and idx < len(class_id):
            item["class_id"] = int(class_id[idx])
        else:
            item["class_id"] = None
        raw.append(item)

    return raw


def main() -> None:
    config = load_config(CONFIG_PATH)

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]

    model_type = str(model_cfg["type"]).lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")

    dataset_path = Path(dataset_cfg["path"])
    coco, val_images = load_validation_images(dataset_path)

    print(f"Config loaded: {CONFIG_PATH}")
    print(f"Dataset path: {dataset_path}")
    print(f"Validation images: {len(val_images)}")
    print(f"Model type: {model_type}")

    model = build_model(model_type)
    prediction_records = []

    for img_info in val_images:
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        image_path = dataset_path / "valid" / "images" / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Validation image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image)

        record = {
            "image_id": image_id,
            "file_name": file_name,
            "detections": detections_to_raw_list(detections),
        }
        prediction_records.append(record)

        print(
            f"Processed image_id={image_id}, file_name={file_name}, "
            f"num_detections={len(detections)}"
        )

    if prediction_records:
        first_record = prediction_records[0]
        print("First prediction record preview:")
        print(json.dumps(first_record, ensure_ascii=False, indent=2))

    output_dir = Path(__file__).resolve().parent.parent / "experiments" / "evaluation_preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions_raw.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prediction_records, f, ensure_ascii=False, indent=2)

    print(f"Prediction records saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Evaluation preview failed: {exc}", file=sys.stderr)
        raise