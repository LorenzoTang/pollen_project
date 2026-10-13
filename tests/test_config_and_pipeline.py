import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.config_utils import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    get_required_path,
    load_config,
    resolve_project_path,
)
from src import run_pipeline
from src.train_rfdetr import prepare_experiment_dir
from src.check_data import resolve_raw_layout, check_raw_data
from src.prepare_rfdetr_dataset import prepare_dataset


class ConfigPathTests(unittest.TestCase):
    def test_new_delivery_can_be_validated_and_prepared(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "数据"
            annotation_dir = raw / "coco - 副本"
            annotation_dir.mkdir(parents=True)
            for name in ("樟.jpg", "楝.jpg"):
                Image.new("RGB", (32, 32)).save(raw / name)
            data = {
                "images": [
                    {"id": index, "file_name": f"C:\\old\\{name}", "width": 32, "height": 32}
                    for index, name in enumerate(("樟.jpg", "楝.jpg"), 1)
                ],
                "annotations": [
                    {"id": index, "image_id": index, "category_id": 0, "bbox": [1, 1, 10, 10]}
                    for index in (1, 2)
                ],
                "categories": [{"id": 0, "name": "pollen"}],
            }
            annotation = annotation_dir / "dataset.json"
            annotation.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(resolve_raw_layout(raw), (annotation, raw))
            self.assertEqual(check_raw_data(raw)["status"], "passed")
            output = root / "processed"
            prepare_dataset(raw, output, train_ratio=0.5)
            for split in ("train", "valid"):
                result = json.loads((output / split / "_annotations.coco.json").read_text(encoding="utf-8"))
                self.assertEqual(len(result["images"]), 1)
                self.assertEqual(result["categories"][0]["id"], 1)
                self.assertTrue((output / split / result["images"][0]["file_name"]).is_file())

    def test_experiment_directory_cannot_overwrite_previous_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text("model: {}\n", encoding="utf-8")
            config = {"experiment": {"root": str(root / "runs")}}
            experiment = prepare_experiment_dir(config, config_path, "existing")
            original = (experiment / "config.yaml").read_bytes()
            config_path.write_text("changed: true\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_experiment_dir(config, config_path, "existing")
            self.assertEqual((experiment / "config.yaml").read_bytes(), original)

    def test_baseline_config_uses_one_processed_dataset_path(self):
        config = load_config(DEFAULT_CONFIG_PATH)

        processed_root = get_required_path(
            config, "processed_data", "root", PROJECT_ROOT
        )
        dataset_root = get_required_path(config, "dataset", "path", PROJECT_ROOT)

        self.assertEqual(processed_root, dataset_root)
        self.assertEqual(
            get_required_path(config, "experiment", "root", PROJECT_ROOT),
            (PROJECT_ROOT / "experiments").resolve(),
        )

    def test_relative_path_is_resolved_from_project_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            resolved = resolve_project_path("data/processed/example", project_root)

            self.assertEqual(resolved, (project_root / "data/processed/example").resolve())

    def test_required_path_rejects_missing_field(self):
        with self.assertRaisesRegex(ValueError, "dataset.path"):
            get_required_path({"dataset": {}}, "dataset", "path")

    def test_experiment_directory_uses_configured_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text("model: {}\n", encoding="utf-8")
            experiments_root = root / "custom-experiments"
            config = {
                "model": {"name": "RF-DETR-Nano"},
                "experiment": {"root": str(experiments_root)},
            }

            experiment_dir = prepare_experiment_dir(
                config, config_path, experiment_name="fixed-run"
            )

            self.assertEqual(experiment_dir, experiments_root / "fixed-run")
            self.assertTrue((experiment_dir / "checkpoints").is_dir())
            self.assertTrue((experiment_dir / "logs").is_dir())
            self.assertTrue((experiment_dir / "results").is_dir())
            self.assertTrue((experiment_dir / "config.yaml").is_file())


class PipelineControlFlowTests(unittest.TestCase):
    @patch("src.run_pipeline.subprocess.run")
    def test_subprocess_arguments_are_forwarded(self, run_mock):
        run_mock.return_value = subprocess.CompletedProcess([], 0, "", "")
        script = Path("train.py")

        run_pipeline.run_subprocess(script, ["--config", "config.yaml"])

        command = run_mock.call_args.args[0]
        self.assertEqual(command[1:], [str(script), "--config", "config.yaml"])

    def test_configuration_failure_writes_report_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            missing_config = project_root / "missing.yaml"

            with patch.object(run_pipeline, "PROJECT_ROOT", project_root):
                return_code = run_pipeline.main(["--config", str(missing_config)])

            report_path = project_root / "experiments" / "pipeline_report.json"
            self.assertEqual(return_code, 1)
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report.get("error"))


if __name__ == "__main__":
    unittest.main()