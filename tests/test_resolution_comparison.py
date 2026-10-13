import copy
import tempfile
import unittest
from pathlib import Path

from scripts.compare_resolution import compare
from src.training_report import write_json, read_json


class ResolutionComparisonTests(unittest.TestCase):
    def setup_reports(self, root):
        a, d = root / "a", root / "d"
        report = {
            "config": {"training": {"epochs": 50}}, "completed_epochs": 50,
            "identity": {"data_sha256": "same"}, "best_epoch": 40,
            "timing": {"training_seconds": 60, "evaluation_seconds": 2},
            "evaluation": {
                "checkpoint_path": str(a / "checkpoints" / "checkpoint_best_total.pth"),
                "protocol": {"operating_confidence": 0.5, "excluded_background_candidates": 1},
                "metrics": {"mAP50-95": 0.4, "mAP50": 0.5, "mAP75": 0.4,
                            "precision": 0.8, "recall": 0.7, "f1": 0.75, "TP": 7, "FP": 2, "FN": 3},
                "per_class": [{"category_id": 1, "name": "absent", "ground_truth": 0,
                               "AP50-95": None, "AP50": None, "recall": None, "f1": None}]}}
        for path in (a, d):
            write_json(path / "run_status.json", {"status": "success"})
        write_json(a / "results" / "training_report.json", report)
        other = copy.deepcopy(report)
        other["evaluation"]["inference_resolution"] = 640
        other["evaluation"]["checkpoint_path"] = str(d / "checkpoints" / "checkpoint_best_total.pth")
        other["config"]["training"]["resolution"] = 640
        other["evaluation"]["metrics"]["mAP50-95"] = 0.45
        other["evaluation"]["protocol"]["excluded_background_candidates"] = 2
        write_json(d / "results" / "training_report.json", other)
        return a, d

    def test_delta_is_percentage_points_and_missing_class_remains_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, d = self.setup_reports(root)
            result = compare(a, d, root / "output")
            self.assertAlmostEqual(result["metrics"][0]["delta"], 5)
            self.assertIsNone(result["per_class"][0]["delta_pp_AP50-95"])
            self.assertIn("N/A", (root / "output" / "comparison.md").read_text(encoding="utf-8"))

    def test_rejects_changed_data_or_protocol_or_incomplete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for field in ("data", "protocol", "epochs", "config", "checkpoint", "resolution"):
                a, d = self.setup_reports(root)
                path = d / "results" / "training_report.json"
                report = read_json(path)
                if field == "data":
                    report["identity"]["data_sha256"] = "changed"
                elif field == "protocol":
                    report["evaluation"]["protocol"]["operating_confidence"] = 0.3
                elif field == "epochs":
                    report["completed_epochs"] = 1
                elif field == "checkpoint":
                    report["evaluation"]["checkpoint_path"] = str(d / "last.ckpt")
                elif field == "resolution":
                    report["evaluation"]["inference_resolution"] = 384
                else:
                    report["config"]["training"]["batch_size"] = 1
                write_json(path, report)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    compare(a, d, root / "output")


if __name__ == "__main__":
    unittest.main()