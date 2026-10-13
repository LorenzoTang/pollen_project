import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess
import yaml

from pycocotools.coco import COCO

from src.detection_metrics import coco_metrics
from src.training_report import check_identity, epoch_history, read_json, write_json
from src import run_batch
from src.evaluate_rfdetr import resolve_runtime_paths


class DetectionMetricTests(unittest.TestCase):
    def dataset(self):
        coco = COCO()
        coco.dataset = {
            "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "present"}, {"id": 2, "name": "absent"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                             "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}],
        }
        coco.createIndex()
        return coco

    def pred(self, score=0.9, category=1):
        return {"image_id": 1, "category_id": category, "bbox": [0, 0, 10, 10], "score": score}

    def test_ap_is_not_precision_times_recall_and_duplicates_are_fp(self):
        result = coco_metrics(self.dataset(), [self.pred(), self.pred(0.8)])
        self.assertAlmostEqual(result["metrics"]["mAP50-95"], 1.0)
        self.assertEqual(result["metrics"]["precision"], 0.5)
        self.assertEqual(result["metrics"]["recall"], 1.0)
        self.assertEqual(result["metrics"]["FP"], 1)
        self.assertIsNone(result["per_class"][1]["AP50-95"])

    def test_low_score_predictions_affect_ap_but_not_operating_recall(self):
        result = coco_metrics(self.dataset(), [self.pred(0.1)])
        self.assertAlmostEqual(result["metrics"]["mAP50"], 1.0)
        self.assertEqual(result["metrics"]["recall"], 0)
        self.assertEqual(result["metrics"]["FN"], 1)

    def test_empty_predictions_and_wrong_class(self):
        for records in ([], [self.pred(category=2)]):
            result = coco_metrics(self.dataset(), records)
            self.assertEqual(result["metrics"]["mAP50"], 0)
            self.assertEqual(result["metrics"]["FN"], 1)
        self.assertEqual(result["metrics"]["FP"], 1)

    def test_crowd_detections_are_ignored(self):
        coco = self.dataset()
        coco.dataset["annotations"].append({"id": 2, "image_id": 1, "category_id": 1,
                                            "bbox": [30, 30, 20, 20], "area": 400, "iscrowd": 1})
        coco.createIndex()
        pred = self.pred()
        pred["bbox"] = [32, 32, 10, 10]
        result = coco_metrics(coco, [pred])
        self.assertEqual(result["metrics"]["FP"], 0)
        self.assertEqual(result["metrics"]["FN"], 1)


class ReportTests(unittest.TestCase):
    def test_sparse_csv_merges_train_and_val_rows_without_zero_filling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            path.write_text("epoch,step,train/loss,val/loss\n0,1,,3\n0,1,4,\n1,2,2,\n", encoding="utf-8")
            result = epoch_history(path)
            self.assertEqual(result[0], {"epoch": 1, "step": 1, "train/loss": 4, "val/loss": 3})
            self.assertNotIn("val/loss", result[1])

    def test_identity_rejects_modified_config_or_data(self):
        identity = {"config_sha256": "a", "data_sha256": "b"}
        for key in identity:
            with self.assertRaises(ValueError):
                check_identity({**identity, key: "changed"}, identity)

    def test_checkpoint_selection_ignores_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = root / "run" / "checkpoints"
            checkpoints.mkdir(parents=True)
            best = checkpoints / "checkpoint_best_total.pth"
            best.touch()
            (checkpoints / "checkpoint_latest.pth").touch()
            config = {"dataset": {"path": str(root / "data")}}
            paths = resolve_runtime_paths(config, root, root / "run")
            self.assertIn(best, paths)


class QueueTests(unittest.TestCase):
    def setup_queue(self, root):
        config = {"training": {"epochs": 1, "learning_rate": 0.0001, "grad_accum_steps": 1},
                  "experiment": {"root": str(root / "runs")}}
        base = root / "base.yaml"
        base.write_text(yaml.safe_dump(config), encoding="utf-8")
        manifest = root / "batch.yaml"
        manifest.write_text(yaml.safe_dump({"base_config": str(base), "queue_dir": str(root / "queue"),
                                            "tasks": [{"name": "one"}, {"name": "two"}]}), encoding="utf-8")
        return manifest

    def fake_train(self, command, **kwargs):
        if "--report-only" in command:
            experiment = Path(command[-1])
        else:
            config = run_batch.load_config(Path(command[command.index("--config") + 1]))
            experiment = Path(config["experiment"]["root"]) / command[command.index("--experiment-name") + 1]
        write_json(experiment / "results" / "training_report.json", {
            "identity": {"config_sha256": "config", "data_sha256": "data"},
            "completed_epochs": 1, "best_epoch": 1, "last_epoch_metrics": {"train/loss": 1},
            "evaluation": {"metrics": {"mAP50": 0.3}}})
        return subprocess.CompletedProcess(command, 0)

    @patch.object(run_batch, "preflight_check")
    @patch.object(run_batch, "run_identity", return_value={"config_sha256": "config", "data_sha256": "data"})
    def test_queue_runs_in_order_and_skips_success(self, identity, preflight):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.setup_queue(Path(tmp))
            with patch.object(run_batch.subprocess, "run", side_effect=self.fake_train) as process:
                self.assertEqual(run_batch.main(["--manifest", str(manifest)]), 0)
                self.assertEqual(process.call_count, 2)
                self.assertIn("-one-", process.call_args_list[0].args[0][-1])
                self.assertIn("-two-", process.call_args_list[1].args[0][-1])
                self.assertEqual(run_batch.main(["--manifest", str(manifest)]), 0)
                self.assertEqual(process.call_count, 2)

    @patch.object(run_batch, "preflight_check")
    @patch.object(run_batch, "run_identity", return_value={"config_sha256": "config", "data_sha256": "data"})
    def test_failure_continues_and_report_retry_does_not_train(self, identity, preflight):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.setup_queue(root)
            count = 0
            def process(command, **kwargs):
                nonlocal count
                count += 1
                if count == 1:
                    experiment = root / "runs" / command[-1]
                    write_json(experiment / "run_status.json", {"training_completed": True})
                    return subprocess.CompletedProcess(command, 1)
                return self.fake_train(command, **kwargs)
            with patch.object(run_batch.subprocess, "run", side_effect=process):
                self.assertEqual(run_batch.main(["--manifest", str(manifest)]), 1)
            state = read_json(root / "queue" / "queue_status.json")
            self.assertEqual(state["tasks"]["one"]["status"], "failed")
            self.assertEqual(state["tasks"]["two"]["status"], "success")
            with patch.object(run_batch.subprocess, "run", side_effect=self.fake_train) as process_mock:
                self.assertEqual(run_batch.main(["--manifest", str(manifest), "--resume-failed"]), 0)
                self.assertEqual(process_mock.call_count, 1)
                self.assertIn("--report-only", process_mock.call_args.args[0])

    @patch.object(run_batch, "preflight_check")
    @patch.object(run_batch, "run_identity", return_value={"config_sha256": "config", "data_sha256": "data"})
    def test_dry_run_creates_nothing(self, identity, preflight):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.setup_queue(root)
            with patch.object(run_batch.subprocess, "run") as process:
                self.assertEqual(run_batch.main(["--manifest", str(manifest), "--dry-run"]), 0)
                process.assert_not_called()
            self.assertFalse((root / "queue").exists())


if __name__ == "__main__":
    unittest.main()