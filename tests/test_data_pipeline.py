import json
from pathlib import Path
import tempfile
import unittest

from src.coco_validation import validate_coco_data, validate_coco_file
from src.prepare_rfdetr_dataset import prepare_dataset


class DataPipelineTests(unittest.TestCase):
    def make_raw_dataset(self, root: Path) -> Path:
        raw = root / "raw"
        image_root = raw / "coco" / "images"
        image_root.mkdir(parents=True)
        images = []
        annotations = []
        for image_id, name in enumerate(("c.jpg", "a.jpg", "d.jpg", "b.jpg"), start=1):
            (image_root / name).write_bytes(b"synthetic-image")
            images.append(
                {
                    "id": image_id,
                    "file_name": rf"C:\old\machine\{name}",
                    "width": 20,
                    "height": 10,
                }
            )
            annotations.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": 0 if image_id < 4 else 5,
                    "bbox": [1, 1, 5, 4],
                    "area": 20,
                    "iscrowd": 0,
                }
            )
        data = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 0, "name": "common"}, {"id": 5, "name": "rare"}],
        }
        (raw / "coco" / "dataset.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        return raw

    def test_prepare_is_deterministic_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = self.make_raw_dataset(root)
            first = root / "first"
            second = root / "second"
            summary = prepare_dataset(raw, first, seed=7, decode_images=False)
            prepare_dataset(raw, second, seed=7, decode_images=False)

            manifest_a = json.loads((first / "split_manifest.json").read_text(encoding="utf-8"))
            manifest_b = json.loads((second / "split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_a["splits"], manifest_b["splits"])
            self.assertEqual(summary["train_images"] + summary["valid_images"], 4)
            self.assertEqual(summary["train_category_distribution"]["2"], 1)
            self.assertTrue((first / "train" / "_annotations.coco.json").is_file())
            validate_coco_file(first / "train" / "_annotations.coco.json", first / "train", decode_images=False)

            output_data = json.loads(
                (first / "train" / "_annotations.coco.json").read_text(encoding="utf-8")
            )
            self.assertEqual([category["id"] for category in output_data["categories"]], [1, 2])
            self.assertTrue(all("\\" not in image["file_name"] for image in output_data["images"]))

    def test_existing_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = self.make_raw_dataset(root)
            output = root / "processed"
            prepare_dataset(raw, output, decode_images=False)
            marker = output / "marker.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_dataset(raw, output, decode_images=False)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            prepare_dataset(raw, output, overwrite=True, decode_images=False)
            self.assertFalse(marker.exists())

    def test_validator_rejects_out_of_bounds_bbox(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.jpg").write_bytes(b"synthetic-image")
            data = {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [9, 9, 2, 2]}
                ],
                "categories": [{"id": 1, "name": "pollen"}],
            }
            with self.assertRaisesRegex(ValueError, "outside image bounds"):
                validate_coco_data(data, root, decode_images=False)


if __name__ == "__main__":
    unittest.main()