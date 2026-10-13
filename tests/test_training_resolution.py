import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config_utils import DEFAULT_CONFIG_PATH, load_config, PROJECT_ROOT
from src import train_rfdetr, run_batch
from src.evaluate_rfdetr import build_model as build_evaluation_model
from src.training_report import read_json, write_json


class TrainingResolutionTests(unittest.TestCase):
    def test_evaluation_resolution_is_explicit_and_baseline_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint_best_total.pth"
            checkpoint.touch()
            with patch("rfdetr.RFDETR.from_checkpoint") as load:
                build_evaluation_model("nano", checkpoint, resolution=640)
                load.assert_called_once_with(str(checkpoint), trust_checkpoint=True,
                                             resolution=640, positional_encoding_size=40)
                load.reset_mock()
                build_evaluation_model("nano", checkpoint)
                load.assert_called_once_with(str(checkpoint), trust_checkpoint=True)

    def test_formal_experiment_changes_only_resolution(self):
        baseline = load_config(DEFAULT_CONFIG_PATH)
        manifest = load_config(PROJECT_ROOT / "configs" / "batch_resolution640.yaml")
        _, actual = run_batch.task_configs(manifest)[0]
        expected = copy.deepcopy(baseline)
        expected["training"]["resolution"] = 640
        self.assertEqual(actual, expected)

    def test_resolution_reaches_sdk_and_manifest_without_changing_defaults(self):
        for resolution in (None, 640):
            with self.subTest(resolution=resolution), tempfile.TemporaryDirectory() as tmp:
                config = load_config(DEFAULT_CONFIG_PATH)
                if resolution is not None:
                    config["training"]["resolution"] = resolution
                experiment = Path(tmp)
                write_json(experiment / "run_manifest.json", {})
                with patch.object(train_rfdetr, "build_model") as build:
                    train_rfdetr.execute_training(config, {}, {}, experiment)
                arguments = build.return_value.train.call_args.kwargs
                self.assertEqual(arguments["batch_size"], 2)
                self.assertEqual(arguments["grad_accum_steps"], 1)
                self.assertEqual(arguments["lr"], 0.0001)
                self.assertEqual(arguments["seed"], 42)
                if resolution is None:
                    self.assertNotIn("resolution", arguments)
                else:
                    self.assertEqual(arguments["resolution"], resolution)
                self.assertEqual(read_json(experiment / "run_manifest.json")["effective_train_arguments"], arguments)

    def test_preflight_rejects_invalid_resolution(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        with patch.object(train_rfdetr, "validate_coco_file", return_value={"categories": 15}):
            for value in (True, 0, -32, 641, 640.0, "640"):
                with self.subTest(value=value):
                    config["training"]["resolution"] = value
                    with self.assertRaisesRegex(ValueError, "training.resolution"):
                        train_rfdetr.preflight_check(config, DEFAULT_CONFIG_PATH)


if __name__ == "__main__":
    unittest.main()