from pathlib import Path
import json
import sys
from typing import Any, Dict
from collections import defaultdict
from datetime import datetime

import yaml
from PIL import Image
from rfdetr import RFDETRMedium, RFDETRNano
from pycocotools.coco import COCO


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "rfdetr_baseline.yaml"


SUPPORTED_MODEL_TYPES = {"nano", "medium"}

IOU_THRESHOLDS = [round(x / 100, 2) for x in range(50, 100, 5)]


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


def resolve_runtime_paths(config: Dict[str, Any], project_root: Path):
    evaluation_cfg = config.get("evaluation", {})
    training_cfg = config.get("training", {})

    experiment_dir = evaluation_cfg.get("experiment_dir") or training_cfg.get("experiment_dir")
    if experiment_dir:
        experiment_dir = Path(experiment_dir)
    else:
        experiment_dir = find_latest_experiment_dir(project_root)

    checkpoint_path = evaluation_cfg.get("checkpoint_path") or training_cfg.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
    elif experiment_dir is not None:
        ckpt_dir = experiment_dir / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime) if ckpt_dir.exists() else []
        checkpoint_path = ckpts[-1] if ckpts else None
    else:
        checkpoint_path = None

    results_dir = evaluation_cfg.get("results_dir")
    if results_dir:
        results_dir = Path(results_dir)
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
            item["class_id"] = int(class_id[idx])
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
                    "class_id": int(class_id[idx]) if class_id is not None and idx < len(class_id) else None,
                }
            )
    return flat


def evaluate_predictions(coco: COCO, predictions_by_image):
    gt_by_image = build_ground_truth(coco)
    flat_predictions = extract_predictions(predictions_by_image)

    total_gt = sum(len(v) for v in gt_by_image.values())
    metrics = {}

    for thr in IOU_THRESHOLDS:
        preds = sorted(flat_predictions, key=lambda x: x["confidence"], reverse=True)
        matched = set()
        tp = 0
        fp = 0
        for pred_idx, pred in enumerate(preds):
            candidates = gt_by_image.get(pred["image_id"], [])
            best_iou = 0.0
            best_gt_idx = None
            for gt_idx, gt in enumerate(candidates):
                key = (pred["image_id"], gt_idx)
                if key in matched:
                    continue
                if pred["class_id"] is not None and gt["class_id"] != pred["class_id"]:
                    continue
                iou = xyxy_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= thr and best_gt_idx is not None:
                matched.add((pred["image_id"], best_gt_idx))
                tp += 1
            else:
                fp += 1

        fn = max(0, total_gt - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        ap = precision * recall
        metrics[thr] = {
            "ap": ap,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    mAP50 = metrics[0.5]["ap"]
    mAP5095 = sum(v["ap"] for v in metrics.values()) / len(metrics) if metrics else 0.0
    precision = metrics[0.5]["precision"]
    recall = metrics[0.5]["recall"]
    f1 = metrics[0.5]["f1"]

    return {
        "mAP50": mAP50,
        "mAP50-95": mAP5095,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    config = load_config(CONFIG_PATH)

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    project_root = Path(__file__).resolve().parent.parent
    experiment_dir, checkpoint_path, results_dir = resolve_runtime_paths(config, project_root)

    model_type = str(model_cfg["type"]).lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")

    dataset_path = Path(dataset_cfg["path"])
    coco, val_images = load_validation_images(dataset_path)

    print(f"Config loaded: {CONFIG_PATH}")
    print(f"Dataset path: {dataset_path}")
    print(f"Validation images: {len(val_images)}")
    print(f"Model type: {model_type}")
    if experiment_dir is not None:
        print(f"Experiment dir: {experiment_dir}")
    if checkpoint_path is not None:
        print(f"Checkpoint path: {checkpoint_path}")

    model = build_model(model_type)
    prediction_records = []
    predictions_by_image = {}

    for img_info in val_images:
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        image_path = resolve_validation_image_path(dataset_path, file_name)

        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image)
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

    if prediction_records:
        first_record = prediction_records[0]
        print("First prediction record preview:")
        print(json.dumps(first_record, ensure_ascii=False, indent=2))

    results_dir.mkdir(parents=True, exist_ok=True)
    preview_path = results_dir / "predictions_raw.json"
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(prediction_records, f, ensure_ascii=False, indent=2)

    metrics = evaluate_predictions(coco, predictions_by_image)
    evaluation_report = {
        "model": model_type,
        "dataset": str(dataset_path),
        "split": "valid",
        "metrics": metrics,
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