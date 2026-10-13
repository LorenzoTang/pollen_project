"""Compare completed resolution experiments using their independent COCO reports."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training_report import read_json, write_csv, write_json


def compare(baseline, candidate, output):
    reports = []
    for path in (baseline, candidate):
        if read_json(path / "run_status.json")["status"] != "success":
            raise ValueError(f"Experiment has not succeeded: {path}")
        report = read_json(path / "results" / "training_report.json")
        if report["completed_epochs"] != 50:
            raise ValueError("Comparison requires 50 completed epochs")
        if Path(report["evaluation"]["checkpoint_path"]).name != "checkpoint_best_total.pth":
            raise ValueError("Comparison requires checkpoint_best_total.pth")
        reports.append(report)
    a, d = reports
    if d["evaluation"].get("inference_resolution") != 640:
        raise ValueError("Candidate must be independently evaluated at resolution 640")
    expected_config = copy.deepcopy(a["config"])
    expected_config["training"]["resolution"] = 640
    if d["config"] != expected_config:
        raise ValueError("Expected only training.resolution=640 to differ from baseline")
    if a["identity"]["data_sha256"] != d["identity"]["data_sha256"]:
        raise ValueError("Dataset hashes differ")
    protocols = [{k: v for k, v in r["evaluation"]["protocol"].items()
                  if k != "excluded_background_candidates"} for r in reports]
    if protocols[0] != protocols[1]:
        raise ValueError("Evaluation protocols differ")
    metrics = []
    for key in ("mAP50-95", "mAP50", "mAP75", "precision", "recall", "f1", "TP", "FP", "FN"):
        av, dv = a["evaluation"]["metrics"][key], d["evaluation"]["metrics"][key]
        count = key in ("TP", "FP", "FN")
        metrics.append({"metric": key, "baseline": av, "candidate": dv,
                        "delta": (dv - av) * (1 if count else 100),
                        "delta_unit": "count" if count else "percentage_points"})
    classes = []
    d_classes = {r["category_id"]: r for r in d["evaluation"]["per_class"]}
    for row in a["evaluation"]["per_class"]:
        other = d_classes[row["category_id"]]
        if (row["name"], row["ground_truth"]) != (other["name"], other["ground_truth"]):
            raise ValueError("Class definitions or supports differ")
        result = {"name": row["name"], "ground_truth": row["ground_truth"]}
        for key in ("AP50-95", "AP50", "recall", "f1"):
            av, dv = row[key], other[key]
            result.update({f"baseline_{key}": av, f"candidate_{key}": dv,
                           f"delta_pp_{key}": None if av is None or dv is None else (dv-av)*100})
        classes.append(result)
    output.mkdir(parents=True, exist_ok=True)
    result = {"baseline": str(baseline), "candidate": str(candidate),
              "data_sha256": a["identity"]["data_sha256"], "protocol": protocols[0],
              "metrics": metrics, "per_class": classes,
              "timing": {"baseline": a["timing"], "candidate": d["timing"]},
              "best_epoch": {"baseline": a["best_epoch"], "candidate": d["best_epoch"]}}
    write_json(output / "comparison.json", result)
    write_csv(output / "comparison.csv", metrics)
    write_csv(output / "per_class_comparison.csv", classes)
    lines = ["# Nano 640 与 A 组对照（50 轮）", "",
             f"- A：`{baseline}`", f"- D：`{candidate}`",
             "- 数据文件 SHA256 与独立 COCO 评估协议一致；训练 76 张，验证 31 张。",
             "- 配置仅新增 training.resolution=640；保留 batch=2、累积=1、lr=1e-4、seed=42。",
             "- SDK per-batch 多尺度训练：A=224–544、D=480–800，步长 32；验证/推理为 384/640。位置编码尺寸由 SDK 自动同步。",
             "- 两组均评估 checkpoint_best_total.pth；P/R/F1 使用 confidence=0.5、IoU=0.5。", "",
             "| 指标 | A | D（640） | 差值 |", "|---|---:|---:|---:|"]
    for row in metrics:
        count = row["delta_unit"] == "count"
        values = [str(row[k]) if count else f"{row[k]*100:.2f}%" for k in ("baseline", "candidate")]
        unit = "个" if count else "个百分点"
        lines.append(f"| {row['metric']} | {values[0]} | {values[1]} | {row['delta']:+.2f} {unit} |")
    lines += ["", "## 分类别表现", "", "| 类别 | 验证框 | A AP50–95 | D AP50–95 | A Recall | D Recall |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in classes:
        values = ["N/A" if row[k] is None else f"{row[k]*100:.2f}%" for k in
                  ("baseline_AP50-95", "candidate_AP50-95", "baseline_recall", "candidate_recall")]
        lines.append(f"| {row['name']} | {row['ground_truth']} | " + " | ".join(values) + " |")
    lines += ["", "## 耗时和限制", ""]
    for label, report in zip(("A", "D"), reports):
        lines.append(f"- {label}：训练 {report['timing']['training_seconds']/60:.2f} 分钟；"
                     f"独立评估 {report['timing']['evaluation_seconds']:.2f} 秒；最佳轮次 {report['best_epoch']}。")
    lines += ["- 耗时包括运行开销，非标准化推理延迟测试；跨日期系统负载也会影响耗时。",
              "- 仅一次随机种子实验，验证集同时用于模型选择，且只覆盖 9/15 类；不能宣称统计显著或全部类别泛化提升。"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline.resolve(), args.candidate.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()