"""Reproducible run identity and reports; never invent missing measurements."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import importlib.metadata
import platform
from pathlib import Path

try:
    from .config_utils import get_required_path
except ImportError:
    from config_utils import get_required_path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_identity(config):
    root = get_required_path(config, "dataset", "path")
    files, splits = {}, {}
    for split in ("train", "valid"):
        annotation = root / split / "_annotations.coco.json"
        data = read_json(annotation)
        files[str(annotation.relative_to(root))] = sha256(annotation)
        counts = {c["name"]: sum(a["category_id"] == c["id"] for a in data["annotations"])
                  for c in data["categories"]}
        splits[split] = {"images": len(data["images"]), "boxes": len(data["annotations"]),
                         "class_counts": counts}
        for image in data["images"]:
            path = root / split / image["file_name"]
            files[str(path.relative_to(root))] = sha256(path)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    data_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"config_sha256": config_hash, "data_sha256": data_hash, "files": files, "splits": splits}


def software_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("rfdetr", "torch", "torchvision", "pytorch-lightning", "pycocotools", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def check_identity(actual, expected):
    for key in ("config_sha256", "data_sha256"):
        if actual[key] != expected[key]:
            raise ValueError(f"Run identity changed ({key}); use a new experiment/queue")


def epoch_history(path):
    epochs = {}
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if not row.get("epoch"):
                continue
            epoch = int(float(row["epoch"])) + 1
            target = epochs.setdefault(epoch, {"epoch": epoch})
            for key, value in row.items():
                if key == "epoch" or value in (None, ""):
                    continue
                number = float(value)
                target[key] = number if math.isfinite(number) else None
    return [epochs[key] for key in sorted(epochs)]


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(experiment):
    experiment = Path(experiment)
    manifest = read_json(experiment / "run_manifest.json")
    state = read_json(experiment / "run_status.json")
    evaluation = read_json(experiment / "results" / "evaluation_report.json")
    history = manifest.get("prior_history", []) + epoch_history(experiment / "checkpoints" / "metrics.csv")
    merged = {}
    for row in history:
        merged.setdefault(row["epoch"], {}).update(row)
    history = [merged[key] for key in sorted(merged)]
    if not history:
        raise ValueError("No epoch metrics available; cannot produce a complete training report")
    best_key = "val/ema_mAP_50_95" if any("val/ema_mAP_50_95" in r for r in history) else "val/mAP_50_95"
    candidates = [r for r in history if r.get(best_key) is not None]
    best = max(candidates, key=lambda r: r[best_key]) if candidates else None
    warnings = ["结果来自用于模型选择的验证集；没有独立测试集，不能作为最终泛化性能结论。",
                "目标检测无可枚举的真负例，accuracy 不适用；请使用 AP、precision、recall 和 F1。",
                "训练器指标与独立 COCOeval 的候选框/阈值约定可能不同；跨实验比较采用独立 COCOeval。",
                "损失为训练器记录值；不同损失分量的加权方式不同，不能直接相加解释总损失。"]
    missing = [r["name"] for r in evaluation["per_class"] if not r["ground_truth"]]
    if missing:
        warnings.append("验证集无有效标注的类别（AP/recall/F1 为 N/A）：" + "、".join(missing))
    if any(r.get("val/loss") is None for r in history):
        warnings.append("部分轮次未记录验证损失；缺失值不补零或推算。")
    report = {"experiment": experiment.name, "status": "success", "config": manifest["config"],
              "identity": manifest["identity"], "environment": manifest["environment"],
              "requested_epochs": manifest["config"]["training"]["epochs"],
              "completed_epochs": max(r["epoch"] for r in history), "recorded_epochs": len(history),
              "best_epoch": best["epoch"] if best else None, "best_epoch_metric": best_key,
              "best_epoch_metrics": best, "last_epoch_metrics": history[-1], "history": history,
              "timing": {key: state[key] for key in ("started_at", "training_seconds", "evaluation_seconds") if key in state},
              "effective_train_arguments": manifest.get("effective_train_arguments", {}),
              "evaluation": evaluation, "warnings": warnings,
              "checkpoint_sha256": sha256(evaluation["checkpoint_path"])}
    results = experiment / "results"
    write_json(results / "training_report.json", report)
    write_csv(results / "epoch_metrics.csv", history)
    write_csv(results / "per_class_metrics.csv", evaluation["per_class"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, keys, title in (
        (axes[0], ("train/loss", "val/loss"), "Epoch losses"),
        (axes[1], ("val/mAP_50", "val/mAP_50_95", "val/F1"), "Trainer validation metrics"),
    ):
        for key in keys:
            axis.plot([r["epoch"] for r in history], [r.get(key, float("nan")) for r in history], marker="o", label=key)
        axis.set(title=title, xlabel="Epoch (1-based)")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(results / "training_curves.png", dpi=160)
    plt.close(figure)
    def fmt(value):
        return "N/A" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)
    def table(rows, fields):
        return ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"] + [
            "| " + " | ".join(fmt(row.get(key)) for key in fields) + " |" for row in rows]
    lines = [f"# 训练报告：{experiment.name}", "", f"计划轮次：{report['requested_epochs']}；已完成轮次：{report['completed_epochs']}；最佳轮次：{report['best_epoch']}（从 1 开始）。",
             f"本次训练耗时：{fmt(state.get('training_seconds'))} 秒；评估耗时：{fmt(state.get('evaluation_seconds'))} 秒。",
             f"最佳权重：{evaluation['checkpoint_path']}", "", "## 数据与复现", "",
             f"配置 SHA256：{manifest['identity']['config_sha256']}",
             f"数据 SHA256：{manifest['identity']['data_sha256']}", "",
             "```json", json.dumps({"config": manifest["config"], "environment": manifest["environment"],
                                    "effective_train_arguments": manifest.get("effective_train_arguments", {}),
                                    "splits": manifest["identity"]["splits"]}, ensure_ascii=False, indent=2), "```",
             "", "## 最佳权重独立评估", "",
             "AP：IoU 0.50–0.95，101 点插值，maxDets=[1,10,100]。小/中/大目标按 COCO 面积范围计算。",
             f"P/R/F1：置信度 ≥ {evaluation['protocol']['operating_confidence']}，IoU ≥ {evaluation['protocol']['operating_iou']}，按微平均汇总；TP/FP/FN 为框数量。", ""]
    lines += table([{"metric": k, "value": v} for k, v in evaluation["metrics"].items()], ["metric", "value"])
    lines += ["", "## 分类别结果", ""] + table(evaluation["per_class"], list(evaluation["per_class"][0]))
    lines += ["", "## 逐轮指标", "", "![训练曲线](training_curves.png)", "", "以下是关键指标；完整损失分量、辅助损失、诊断项及所有训练器指标见 epoch_metrics.csv 与 training_report.json。", ""]
    lines += table(history, ["epoch", "train/loss", "val/loss", "train/loss_ce", "train/loss_bbox", "train/loss_giou", "val/mAP_50", "val/mAP_50_95", "val/precision", "val/recall", "val/F1"])
    lines += ["", "## 解释与限制", ""] + [f"- {warning}" for warning in warnings]
    (results / "training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Training report: {results / 'training_report.md'}")
    print(json.dumps({"completed_epochs": report["completed_epochs"], "best_epoch": report["best_epoch"],
                      "last_train_loss": history[-1].get("train/loss"), "last_val_loss": history[-1].get("val/loss"),
                      **evaluation["metrics"]}, ensure_ascii=False, indent=2))
    return report