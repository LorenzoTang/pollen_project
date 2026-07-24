from __future__ import annotations

import json
from datetime import datetime
import subprocess
from pathlib import Path
import sys
from typing import Any, Dict

import yaml

try:
    from .check_data import check_raw_data
    from .prepare_rfdetr_dataset import prepare_dataset
except ImportError:  # pragma: no cover
    from check_data import check_raw_data
    from prepare_rfdetr_dataset import prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rfdetr_baseline.yaml"
DEFAULT_PIPELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def log(message: str) -> None:
    print(message)


def validate_coco_dataset(processed_root: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "status": "success",
        "processed_root": str(processed_root),
        "splits": {},
    }

    for split in ["train", "valid"]:
        ann_path = processed_root / split / "_annotations.coco.json"
        img_dir = processed_root / split

        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        image_files = {img["file_name"] for img in data.get("images", [])}
        missing_images = [name for name in image_files if not (img_dir / name).exists()]

        split_report = {
            "images": len(data.get("images", [])),
            "annotations": len(data.get("annotations", [])),
            "categories": len(data.get("categories", [])),
            "missing_images": missing_images,
        }

        if missing_images:
            raise FileNotFoundError(f"Missing images in {split}: {missing_images[:5]}")

        report["splits"][split] = split_report

    return report


def write_pipeline_report(report: Dict[str, Any], experiments_root: Path) -> Path:
    experiments_root.mkdir(parents=True, exist_ok=True)
    report_path = experiments_root / "pipeline_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path


def latest_experiment_dir(experiments_root: Path) -> Path | None:
    candidates = [p for p in experiments_root.glob("RF-DETR-*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_subprocess(script_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )


def build_artifacts(experiment_dir: Path | None) -> Dict[str, str]:
    if experiment_dir is None:
        return {"experiment_dir": "", "checkpoint": "", "logs": ""}

    checkpoint_dir = experiment_dir / "checkpoints"
    logs_dir = experiment_dir / "logs"

    checkpoint = ""
    if checkpoint_dir.exists():
        checkpoint_candidates = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if checkpoint_candidates:
            checkpoint = str(checkpoint_candidates[-1])

    return {
        "experiment_dir": str(experiment_dir),
        "checkpoint": checkpoint,
        "logs": str(logs_dir),
    }


def load_evaluation_report(experiment_dir: Path | None) -> Dict[str, Any]:
    if experiment_dir is None:
        return {"status": "failed", "error": "evaluation report not found"}

    report_path = experiment_dir / "results" / "evaluation_report.json"
    if not report_path.exists():
        return {"status": "failed", "error": "evaluation report not found"}

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    return {
        "status": "success",
        "report_path": str(report_path),
        "metrics": report.get("metrics", {}),
    }


def run_stage(stage_name: str, stage_func):
    log(stage_name)
    return stage_func()


def main() -> None:
    config_path = DEFAULT_PIPELINE_CONFIG_PATH if DEFAULT_PIPELINE_CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH
    config = load_config(config_path)

    raw_root = Path(config.get("raw_data", {}).get("root", PROJECT_ROOT / "data" / "raw" / "标签数据(1)"))
    processed_root = Path(config.get("processed_data", {}).get("root", PROJECT_ROOT / "data" / "processed" / "rfdetr_pollen"))
    experiments_root = Path(config.get("experiment", {}).get("root", PROJECT_ROOT / "experiments"))
    do_evaluate = bool(config.get("steps", {}).get("evaluate", False))

    if not raw_root.exists():
        raise FileNotFoundError(f"Configured raw dataset path does not exist: {raw_root}")

    processed_parent = processed_root.parent
    if not processed_parent.exists():
        raise FileNotFoundError(f"Configured processed output parent path does not exist: {processed_parent}")

    pipeline_report: Dict[str, Any] = {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "config": str(config_path),
        "stages": {},
        "artifacts": {"experiment_dir": "", "checkpoint": "", "logs": ""},
    }

    print("[PIPELINE] Starting pipeline")

    try:
        print("[1/5] Raw data check")
        raw_report = check_raw_data(raw_root)
        pipeline_report["stages"]["raw_check"] = {"status": "passed", "report": raw_report}
        print("PASS")

        print("[2/5] Dataset preparation")
        prep_report = prepare_dataset(raw_base=raw_root, output_root=processed_root)
        pipeline_report["stages"]["dataset_prepare"] = {"status": "passed", "report": prep_report}
        print("PASS")

        print("[3/5] COCO validation")
        coco_report = validate_coco_dataset(processed_root)
        pipeline_report["stages"]["validation"] = {"status": "passed", "report": coco_report}
        print("PASS")

        print("[4/5] Training")
        train_script = PROJECT_ROOT / "src" / "train_rfdetr.py"
        train_result = run_subprocess(train_script)
        training_stage: Dict[str, Any] = {
            "status": "passed" if train_result.returncode == 0 else "failed",
            "returncode": train_result.returncode,
            "stdout": train_result.stdout,
            "stderr": train_result.stderr,
        }
        pipeline_report["stages"]["training"] = training_stage
        if train_result.returncode != 0:
            raise RuntimeError("training stage failed")

        experiment_dir = latest_experiment_dir(experiments_root)
        pipeline_report["artifacts"] = build_artifacts(experiment_dir)

        print("[5/5] Evaluation")
        if do_evaluate:
            eval_script = PROJECT_ROOT / "src" / "evaluate_rfdetr.py"
            eval_result = run_subprocess(eval_script)
            evaluation_stage: Dict[str, Any] = {
                "status": "passed" if eval_result.returncode == 0 else "failed",
                "returncode": eval_result.returncode,
                "stdout": eval_result.stdout,
                "stderr": eval_result.stderr,
            }
            if eval_result.returncode != 0:
                raise RuntimeError("evaluation stage failed")
            evaluation_report = load_evaluation_report(experiment_dir)
            evaluation_stage["report_load_status"] = evaluation_report.get("status", "failed")
            if evaluation_report.get("status") == "success":
                evaluation_stage["report_path"] = evaluation_report.get("report_path", "")
                evaluation_stage["metrics"] = evaluation_report.get("metrics", {})
            else:
                evaluation_stage["error"] = evaluation_report.get("error", "evaluation report not found")
            pipeline_report["stages"]["evaluation"] = evaluation_stage
        else:
            print("SKIPPED")
            pipeline_report["stages"]["evaluation"] = {"status": "skipped"}

        pipeline_report["status"] = "success"
    except Exception as exc:
        print("[PIPELINE] Stage failed: pipeline execution")
        print("Error:")
        print(exc)
        print("Pipeline stopped.")
        pipeline_report["status"] = "failed"
        pipeline_report["error"] = str(exc)

    report_path = write_pipeline_report(pipeline_report, experiments_root)
    print(f"Pipeline report saved to: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Pipeline execution failed: {exc}", file=sys.stderr)
        raise