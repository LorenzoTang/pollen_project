import argparse
from pathlib import Path
from datetime import datetime
import shutil
import sys
import gc
import os
import subprocess
import time
from typing import Any, Dict

try:
    from .config_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT, get_required_path, load_config
    from .training_report import check_identity, epoch_history, generate_report, read_json, run_identity, sha256, software_versions, write_json
except ImportError:  # pragma: no cover - supports direct script execution
    from config_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT, get_required_path, load_config
    from training_report import check_identity, epoch_history, generate_report, read_json, run_identity, sha256, software_versions, write_json

try:
    from .coco_validation import validate_coco_file
except ImportError:  # pragma: no cover
    from coco_validation import validate_coco_file


SUPPORTED_MODEL_TYPES = {"nano", "medium"}


def prepare_experiment_dir(
    config: Dict[str, Any],
    config_path: Path,
    experiment_name: str | None = None,
) -> Path:
    model_cfg = config.get("model", {})
    experiments_root = get_required_path(config, "experiment", "root", PROJECT_ROOT)
    experiment_name = experiment_name or config.get("experiment_name")

    if not experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_cfg.get("name", "experiment")
        experiment_name = f"{model_name}_{timestamp}"

    if Path(experiment_name).name != experiment_name or experiment_name in {".", ".."}:
        raise ValueError("Experiment name must be a directory name without path separators")
    experiment_dir = experiments_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=False)

    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "results").mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, experiment_dir / "config.yaml")

    return experiment_dir


def write_log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def preflight_check(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    required_top_keys = ("model", "dataset", "training", "experiment")
    for key in required_top_keys:
        if key not in config:
            raise ValueError(f"Config missing required section: {key}")

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    if "type" not in model_cfg:
        raise ValueError("Config missing required field: model.type")

    model_type = str(model_cfg["type"]).lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: nano, medium")

    if "path" not in dataset_cfg:
        raise ValueError("Config missing required field: dataset.path")

    dataset_path = get_required_path(config, "dataset", "path", PROJECT_ROOT)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    train_ann = dataset_path / "train" / "_annotations.coco.json"
    val_ann = dataset_path / "valid" / "_annotations.coco.json"
    if not train_ann.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {train_ann}")
    if not val_ann.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {val_ann}")

    train_report = validate_coco_file(train_ann, dataset_path / "train")
    val_report = validate_coco_file(val_ann, dataset_path / "valid")
    expected_classes = int(dataset_cfg.get("classes", 0))
    if expected_classes and train_report["categories"] != expected_classes:
        raise ValueError(
            f"dataset.classes={expected_classes}, but COCO has {train_report['categories']} categories"
        )
    if val_report["categories"] != train_report["categories"]:
        raise ValueError("Train and valid category counts differ")

    for field in ("epochs", "batch_size", "grad_accum_steps"):
        if int(training_cfg.get(field, 0)) < 1:
            raise ValueError(f"training.{field} must be >= 1")
    if float(training_cfg.get("learning_rate", 0)) <= 0:
        raise ValueError("training.learning_rate must be > 0")
    if "resolution" in training_cfg:
        resolution = training_cfg["resolution"]
        # Both supported models use patch_size=16 and num_windows=2.
        if type(resolution) is not int or resolution <= 0 or resolution % 32:
            raise ValueError("training.resolution must be a positive integer multiple of 32")

    get_required_path(config, "experiment", "root", PROJECT_ROOT).mkdir(parents=True, exist_ok=True)

    print("Preflight check passed.")
    return {"train": train_report, "valid": val_report}


def build_model(model_type: str):
    try:
        from rfdetr import RFDETRMedium, RFDETRNano
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR is not installed. Install project dependencies before training."
        ) from exc

    model_type = model_type.lower()

    if model_type == "nano":
        return RFDETRNano()

    if model_type == "medium":
        return RFDETRMedium()

    raise ValueError(f"Unsupported model type: {model_type}")


def check_training_environment(require_cuda: bool = True) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Run scripts/setup_windows_cuda.ps1 before training."
        ) from exc
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by this configuration, but PyTorch cannot access the GPU. "
            "Run src/check_environment.py --require-cuda for details."
        )
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an RF-DETR pollen detector.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-from", type=Path, help="Resume an incomplete run into a NEW experiment directory.")
    parser.add_argument("--report-only", type=Path, help="Retry evaluation/report for a trained run without retraining.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.report_only:
        experiment_dir = args.report_only.resolve()
        manifest = read_json(experiment_dir / "run_manifest.json")
        check_identity(run_identity(manifest["config"]), manifest["identity"])
        state = read_json(experiment_dir / "run_status.json")
        if not state.get("training_completed"):
            raise ValueError("Training did not finish; use --resume-from instead")
        finish_report(experiment_dir, state)
        return
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    validation_report = preflight_check(config, config_path)
    if args.preflight_only:
        print("Preflight-only mode: model construction and training skipped.")
        print(validation_report)
        return
    environment_report = check_training_environment(
        bool(config.get("training", {}).get("require_cuda", True))
    )
    environment_report["versions"] = software_versions()
    environment_report["source_sha256"] = {path.name: sha256(path) for path in (PROJECT_ROOT / "src").glob("*.py")}
    if config["training"].get("resume"):
        raise ValueError("Use --resume-from for identity-checked full-state recovery")
    identity = run_identity(config)
    prior_history = []
    resume_checkpoint = None
    if args.resume_from:
        previous = args.resume_from.resolve()
        old_manifest = read_json(previous / "run_manifest.json")
        check_identity(identity, old_manifest["identity"])
        if read_json(previous / "run_status.json").get("training_completed"):
            raise ValueError("Training already completed; use --report-only")
        resume_checkpoint = previous / "checkpoints" / "last.ckpt"
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        prior_history = old_manifest.get("prior_history", []) + epoch_history(previous / "checkpoints" / "metrics.csv")
    experiment_dir = prepare_experiment_dir(config, config_path, args.experiment_name)
    # Preserve the old best export: a resumed run may never improve on it.
    if args.resume_from:
        for checkpoint in (previous / "checkpoints").glob("checkpoint_best*.pth"):
            shutil.copy2(checkpoint, experiment_dir / "checkpoints" / checkpoint.name)
    write_json(experiment_dir / "run_manifest.json", {
        "config": config, "identity": identity, "environment": environment_report,
        "started_at": datetime.now().isoformat(), "resume_from": str(args.resume_from) if args.resume_from else None,
        "prior_history": prior_history,
    })
    state = {"status": "training", "training_completed": False, "started_at": datetime.now().isoformat()}
    write_json(experiment_dir / "run_status.json", state)
    started = time.perf_counter()
    try:
        execute_training(config, validation_report, environment_report, experiment_dir, resume_checkpoint)
        state.update(training_completed=True, training_seconds=time.perf_counter() - started)
        finish_report(experiment_dir, state)
    except BaseException as exc:
        state.update(status="failed", error=str(exc), ended_at=datetime.now().isoformat())
        write_json(experiment_dir / "run_status.json", state)
        raise


def finish_report(experiment_dir, state):
    state.update(status="evaluating")
    write_json(experiment_dir / "run_status.json", state)
    started = time.perf_counter()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        with (experiment_dir / "logs" / "evaluation.log").open("a", encoding="utf-8") as stream:
            subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "evaluate_rfdetr.py"),
                            "--config", str(experiment_dir / "config.yaml"),
                            "--experiment-dir", str(experiment_dir), "--checkpoint-path",
                            str(experiment_dir / "checkpoints" / "checkpoint_best_total.pth")],
                           env=env, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        state.update(evaluation_seconds=time.perf_counter() - started, status="reporting")
        write_json(experiment_dir / "run_status.json", state)
        generate_report(experiment_dir)
        state.pop("error", None)
        state.update(status="success", ended_at=datetime.now().isoformat())
        write_json(experiment_dir / "run_status.json", state)
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        write_json(experiment_dir / "run_status.json", state)
        raise


def execute_training(config, validation_report, environment_report, experiment_dir, resume_checkpoint=None):
    model_cfg = config["model"]
    training_cfg = config["training"]
    experiment_name = experiment_dir.name
    dataset_path = get_required_path(config, "dataset", "path", PROJECT_ROOT)
    log_path = experiment_dir / "logs" / "train.log"

    write_log(log_path, "Training started")
    write_log(log_path, f"experiment_name={experiment_name}")
    write_log(log_path, f"model.type={model_cfg['type']}")
    write_log(log_path, f"dataset.path={dataset_path}")
    write_log(log_path, f"epochs={training_cfg['epochs']}")
    write_log(log_path, f"batch_size={training_cfg['batch_size']}")
    write_log(log_path, f"learning_rate={training_cfg['learning_rate']}")
    write_log(log_path, "Preflight check passed")
    write_log(log_path, f"validation={validation_report}")
    write_log(log_path, f"environment={environment_report}")
    write_log(log_path, f"experiment_dir={experiment_dir}")

    model = build_model(model_cfg["type"])

    write_log(log_path, "Begin training")
    print(f"Training logs will be saved to: {log_path}")

    train_kwargs = dict(
        dataset_dir=str(dataset_path),
        epochs=training_cfg["epochs"],
        batch_size=training_cfg["batch_size"],
        grad_accum_steps=training_cfg["grad_accum_steps"],
        lr=training_cfg["learning_rate"],
        output_dir=str(experiment_dir / "checkpoints"),
    )
    # Explicit report-related settings for the installed RF-DETR training API.
    train_kwargs["seed"] = int(config.get("project", {}).get("seed", 42))
    train_kwargs.update(compute_val_loss=True, log_per_class_metrics=True,
                        eval_max_dets=100, eval_interval=1, best_model_metric="map")
    for key in ("num_workers", "device", "amp_dtype", "resolution"):
        if key in training_cfg:
            train_kwargs[key] = training_cfg[key]
    if resume_checkpoint:
        train_kwargs["resume"] = str(resume_checkpoint)
    manifest = read_json(experiment_dir / "run_manifest.json")
    manifest["effective_train_arguments"] = train_kwargs
    write_json(experiment_dir / "run_manifest.json", manifest)
    model.train(**train_kwargs)
    del model
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    write_log(log_path, "Training completed")
    print("Training completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Preflight/Training failed: {exc}", file=sys.stderr)
        raise