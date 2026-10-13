"""Standard COCO bbox AP/AR and fixed-operating-point detection counts."""
from __future__ import annotations

import copy
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_metrics(coco, records, *, confidence=0.5, iou=0.5):
    if not 0 <= confidence <= 1 or not 0 < iou <= 1:
        raise ValueError("Invalid confidence or IoU threshold")
    valid_categories = set(coco.getCatIds())
    for record in records:
        if record["category_id"] not in valid_categories:
            raise ValueError(f"Unknown predicted category: {record['category_id']}")
    if records:
        detections = coco.loadRes(records)
    else:
        detections = COCO()
        detections.dataset = {
            "images": copy.deepcopy(coco.dataset["images"]),
            "categories": copy.deepcopy(coco.dataset["categories"]), "annotations": [],
        }
        detections.createIndex()
    evaluator = COCOeval(coco, detections, "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = ("mAP50-95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large",
            "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large")
    metrics = {key: float(value) if value >= 0 else None for key, value in zip(keys, evaluator.stats)}

    # COCO's matching handles duplicates, crowd/ignore regions and maxDets.
    operating = COCOeval(coco, detections, "bbox")
    operating.params.iouThrs = np.array([iou])
    operating.params.maxDets = [1, 10, 100]
    operating.evaluate()
    per_class = []
    for index, category_id in enumerate(evaluator.params.catIds):
        category = coco.cats[category_id]
        entries = [e for e in operating.evalImgs if e is not None
                   and e["category_id"] == category_id and e["aRng"] == operating.params.areaRng[0]]
        tp = fp = fn = support = 0
        for entry in entries:
            chosen = np.asarray(entry["dtScores"]) >= confidence
            ignore = entry["dtIgnore"][0].astype(bool)
            matched = entry["dtMatches"][0] > 0
            hits = int(np.count_nonzero(chosen & ~ignore & matched))
            truth = int(np.count_nonzero(~entry["gtIgnore"].astype(bool)))
            tp += hits
            fp += int(np.count_nonzero(chosen & ~ignore & ~matched))
            fn += truth - hits
            support += truth
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else None
        f1 = 2 * tp / (2 * tp + fp + fn) if support else None
        ap = evaluator.eval["precision"][:, :, index, 0, -1]
        ap50 = ap[0]
        per_class.append({"category_id": int(category_id), "name": category["name"],
                          "ground_truth": support, "TP": tp, "FP": fp, "FN": fn,
                          "precision": precision, "recall": recall, "f1": f1,
                          "AP50-95": float(ap[ap >= 0].mean()) if (ap >= 0).any() else None,
                          "AP50": float(ap50[ap50 >= 0].mean()) if (ap50 >= 0).any() else None})
    tp, fp, fn = (sum(row[key] for row in per_class) for key in ("TP", "FP", "FN"))
    metrics.update(TP=tp, FP=fp, FN=fn,
                   precision=tp / (tp + fp) if tp + fp else 0.0,
                   recall=tp / (tp + fn) if tp + fn else None,
                   f1=2 * tp / (2 * tp + fp + fn) if tp + fn else None)
    return {"metrics": metrics, "per_class": per_class, "protocol": {
        "implementation": "pycocotools.COCOeval", "iou_type": "bbox", "AP_iou": "0.50:0.05:0.95",
        "max_detections": [1, 10, 100], "operating_confidence": confidence, "operating_iou": iou,
        "precision_recall_aggregation": "micro", "missing_class_AP": None,
        "accuracy": "Not defined: detection has no enumerable true-negative count. Use AP/precision/recall/F1.",
    }}