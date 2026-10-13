from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Any, Dict
from collections import defaultdict
from datetime import datetime

try:
    from .config_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT, get_required_path, load_config, resolve_project_path
    from .detection_metrics import coco_metrics
except ImportError:  # pragma: no cover - supports direct script execution
    from config_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT, get_required_path, load_config, resolve_project_path
    from detection_metrics import coco_metrics


SUPPORTED_MODEL_TYPES = {"nano", "medium"}

IOU_THRESHOLDS = [round(x / 100, 2) for x in range(50, 100, 5)]


def build_model(model_type: str, checkpoint_path: Path | None = None, resolution: int | None = None):
    try:
        from rfdetr import RFDETR, RFDETRMedium, RFDETRNano
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR is not installed. Install project dependencies before evaluation."
        ) from exc

    model_type = model_type.lower()
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        kwargs = {}
        if resolution is not None:
            if type(resolution) is not int or resolution <= 0 or resolution % 32:
                raise ValueError("Evaluation resolution must be a positive integer multiple of 32")
            kwargs["resolution"] = resolution
            kwargs["positional_encoding_size"] = resolution // 16
        return RFDETR.from_checkpoint(str(checkpoint_path), trust_checkpoint=True, **kwargs)
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
    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is not installed. Install project dependencies before evaluation."
        ) from exc

    ann_path = dataset_path / "valid" / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Validation annotation file does not exist: {ann_path}")

    coco = COCO(str(ann_path))
    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)
    return coco, images


def resolve_validation_image_path(dataset_path: Path, file_name: str) -> Path:
    candidates = [
        dataset_path / "valid" / "images" / file_name,
        dataset_path / "valid" / file_name,
    ]
    for image_path in candidates:
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Validation image not found: {candidates[0]}")


def find_latest_experiment_dir(project_root: Path) -> Path | None:
    experiments_root = project_root / "experiments"
    candidates = [p for p in experiments_root.glob("RF-DETR-*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_runtime_paths(
    config: Dict[str, Any],
    project_root: Path,
    experiment_dir_override: Path | None = None,
):
    evaluation_cfg = config.get("evaluation", {})
    training_cfg = config.get("training", {})

    experiment_dir = experiment_dir_override or evaluation_cfg.get("experiment_dir") or training_cfg.get("experiment_dir")
    if experiment_dir:
        experiment_dir = resolve_project_path(experiment_dir, project_root)
    else:
        experiment_dir = find_latest_experiment_dir(project_root)

    checkpoint_path = evaluation_cfg.get("checkpoint_path") or training_cfg.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = resolve_project_path(checkpoint_path, project_root)
    elif experiment_dir is not None:
        ckpt_dir = experiment_dir / "checkpoints"
        checkpoint_path = ckpt_dir / "checkpoint_best_total.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Best checkpoint not found; specify --checkpoint-path: {checkpoint_path}")
    else:
        checkpoint_path = None

    results_dir = evaluation_cfg.get("results_dir")
    if results_dir:
        results_dir = resolve_project_path(results_dir, project_root)
    elif experiment_dir is not None:
        results_dir = experiment_dir / "results"
    else:
        results_dir = project_root / "experiments" / "evaluation_preview"

    return experiment_dir, checkpoint_path, results_dir


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
        bbox_values = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)
        item = {"bbox_xyxy": [float(v) for v in bbox_values]}
        if confidence is not None and idx < len(confidence):
            item["confidence"] = float(confidence[idx])
        else:
            item["confidence"] = None
        if class_id is not None and idx < len(class_id):
            # RF-DETR returns zero-based class indices, while COCO category_id
            # values in this project are one-based.
            item["class_id"] = int(class_id[idx]) + 1
        else:
            item["class_id"] = None
        raw.append(item)

    return raw


def xyxy_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_ground_truth(coco: COCO):
    gt_by_image = defaultdict(list)
    for ann in coco.dataset.get("annotations", []):
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        gt_by_image[ann["image_id"]].append(
            {
                "class_id": ann.get("category_id"),
                "bbox": [x, y, x + w, y + h],
            }
        )
    return gt_by_image


def extract_predictions(predictions_by_image):
    flat = []
    for image_id, detections in predictions_by_image.items():
        xyxy = getattr(detections, "xyxy", None)
        confidence = getattr(detections, "confidence", None)
        class_id = getattr(detections, "class_id", None)
        if xyxy is None:
            continue
        for idx, bbox in enumerate(xyxy):
            flat.append(
                {
                    "image_id": image_id,
                    "bbox": [float(v) for v in bbox.tolist()],
                    "confidence": float(confidence[idx]) if confidence is not None and idx < len(confidence) else 0.0,
                    # Convert the model's zero-based class index to the
                    # one-based COCO category_id used by the annotations.
                    "class_id": int(class_id[idx]) + 1 if class_id is not None and idx < len(class_id) else None,
                }
            )
    return flat


def evaluate_predictions(coco: COCO, predictions_by_image):
    flat_predictions = extract_predictions(predictions_by_image)
    return coco_metrics(coco, to_coco_records(flat_predictions))["metrics"]


def to_coco_records(predictions):
    records = []
    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox"]
        records.append({"image_id": pred["image_id"], "category_id": pred["class_id"],
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": pred["confidence"]})
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate an RF-DETR pollen detector.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Explicit model checkpoint to evaluate; overrides automatic selection.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    experiment_dir, checkpoint_path, results_dir = resolve_runtime_paths(
        config, PROJECT_ROOT, args.experiment_dir
    )
    if args.checkpoint_path is not None:
        checkpoint_path = resolve_project_path(args.checkpoint_path, PROJECT_ROOT)

    model_type = str(model_cfg["type"]).lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")

    dataset_path = get_required_path(config, "dataset", "path", PROJECT_ROOT)
    coco, val_images = load_validation_images(dataset_path)

    print(f"Config loaded: {config_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"Validation images: {len(val_images)}")
    print(f"Model type: {model_type}")
    if experiment_dir is not None:
        print(f"Experiment dir: {experiment_dir}")
    if checkpoint_path is not None:
        print(f"Checkpoint path: {checkpoint_path}")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is not installed. Install project dependencies before evaluation."
        ) from exc

    if checkpoint_path is None:
        raise FileNotFoundError(
            "No trained checkpoint was found; refusing to evaluate an untrained model."
        )
    requested_resolution = config.get("training", {}).get("resolution")
    model = build_model(model_type, checkpoint_path, resolution=requested_resolution)
    actual_resolution = model.model_config.resolution
    if requested_resolution is not None and actual_resolution != requested_resolution:
        raise ValueError("Loaded model resolution does not match the experiment configuration")
    print(f"Inference resolution: {actual_resolution}")
    categories = sorted(coco.dataset["categories"], key=lambda c: c["id"])
    if [c["id"] for c in categories] != list(range(1, len(categories) + 1)):
        raise ValueError("Evaluation requires contiguous COCO category IDs starting at 1")
    if list(model.class_names) != [c["name"] for c in categories]:
        raise ValueError("Checkpoint class names/order do not match evaluation dataset")
    prediction_records = []
    predictions_by_image = {}
    background_candidates = 0

    for img_info in val_images:
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        image_path = resolve_validation_image_path(dataset_path, file_name)

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        # Keep all model candidates for AP; confidence filtering is only for P/R/F1.
        detections = model.predict(image, threshold=0.0)
        # RF-DETR's head has N+1 slots. At zero threshold predict() can expose
        # the unused final background slot; it is not a dataset category.
        labels = detections.class_id
        if ((labels < 0) | (labels > len(categories))).any():
            raise ValueError("Prediction label lies outside the checkpoint detection head")
        background_candidates += int((labels == len(categories)).sum())
        detections = detections[labels < len(categories)]
        predictions_by_image[image_id] = detections

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

    results_dir.mkdir(parents=True, exist_ok=True)
    preview_path = results_dir / "predictions_raw.json"
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(prediction_records, f, ensure_ascii=False, indent=2)

    coco_records = to_coco_records(extract_predictions(predictions_by_image))
    (results_dir / "predictions_coco.json").write_text(
        json.dumps(coco_records, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    eval_config = config.get("evaluation", {})
    evaluation = coco_metrics(coco, coco_records,
                              confidence=float(eval_config.get("confidence_threshold", 0.5)),
                              iou=float(eval_config.get("iou_threshold", 0.5)))
    metrics = evaluation["metrics"]
    evaluation_report = {
        "model": model_type,
        "inference_resolution": actual_resolution,
        "dataset": str(dataset_path),
        "split": "valid",
        "metrics": metrics,
        "per_class": evaluation["per_class"],
        "protocol": {**evaluation["protocol"], "prediction_score_threshold": 0.0,
                     "label_mapping": "model slots 0..N-1 -> COCO IDs 1..N; unused slot N excluded",
                     "excluded_background_candidates": background_candidates},
        "timestamp": datetime.now().isoformat(),
    }
    if experiment_dir is not None:
        evaluation_report["experiment_dir"] = str(experiment_dir)
    if checkpoint_path is not None:
        evaluation_report["checkpoint_path"] = str(checkpoint_path)

    report_path = results_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)

    print(f"Prediction records saved to: {preview_path}")
    print(f"Evaluation report saved to: {report_path}")
    print(json.dumps(evaluation_report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Evaluation preview failed: {exc}", file=sys.stderr)
        raise