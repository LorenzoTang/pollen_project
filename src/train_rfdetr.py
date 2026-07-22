from pathlib import Path
from datetime import datetime
import shutil
from typing import Any, Dict

import yaml
from rfdetr import RFDETRMedium, RFDETRNano


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "rfdetr_baseline.yaml"


def prepare_experiment_dir(config: Dict[str, Any], config_path: Path) -> Path:
    project_root = config_path.resolve().parent.parent
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    output_dir = Path(training_cfg.get("output_dir", project_root / "experiments"))
    experiment_name = config.get("experiment_name")

    if not experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_cfg.get("name", "experiment")
        experiment_name = f"{model_name}_{timestamp}"

    experiments_root = output_dir.parent / "experiments"
    experiment_dir = experiments_root / experiment_name

    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "results").mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, experiment_dir / "config.yaml")

    return experiment_dir


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


def main():
    config = load_config(CONFIG_PATH)
    experiment_dir = prepare_experiment_dir(config, CONFIG_PATH)

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    model = build_model(model_cfg["type"])

    model.train(
        dataset_dir=dataset_cfg["path"],
        epochs=training_cfg["epochs"],
        batch_size=training_cfg["batch_size"],
        grad_accum_steps=training_cfg["grad_accum_steps"],
        lr=training_cfg["learning_rate"],
        output_dir=str(experiment_dir / "checkpoints")
    )


if __name__ == "__main__":
    main()