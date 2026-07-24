import json
from pathlib import Path
from collections import Counter


DEFAULT_ROOT = Path(r"data/raw/标签数据(1)")


def _load_coco_json(raw_root: Path, split: str):
    candidates = [
        raw_root / "annotations" / f"{split}.json",
        raw_root / "coco" / "dataset.json",
    ]

    for json_path in candidates:
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                return json.load(f), json_path

    raise FileNotFoundError(
        f"No COCO annotation file found for split={split}. Tried: {candidates}"
    )


def check_raw_data(raw_root):
    raw_root = Path(raw_root)
    report = {
        "raw_root": str(raw_root),
        "status": "passed",
        "splits": {},
    }

    for split in ["train", "val"]:

        split_report = {
            "status": "passed",
            "annotation_file": None,
            "image_count_annotation": 0,
            "image_count_files": 0,
            "annotation_count": 0,
            "category_count": 0,
            "category_distribution": {},
            "categories": [],
            "missing_images": [],
        }

        print("================")
        print(split)

        data, json_path = _load_coco_json(raw_root, split)
        split_report["annotation_file"] = str(json_path)

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        image_root_candidates = [
            raw_root / split / "images",
            raw_root / "images",
            raw_root / "label",
            raw_root,
        ]

        existing_roots = [p for p in image_root_candidates if p.exists()]
        image_files = []
        missing_images = []

        for img in images:
            file_name = img.get("file_name")
            if not file_name:
                continue
            normalized = Path(file_name).name
            image_files.append(normalized)

            found = False
            for image_root in existing_roots:
                if (image_root / normalized).exists():
                    found = True
                    break
            if not found:
                missing_images.append(normalized)

        image_count_annotation = len(images)
        image_count_files = len(set(image_files)) - len(set(missing_images))
        annotation_count = len(annotations)
        category_count = len(categories)

        print("annotation images:", image_count_annotation)
        print("real image files:", image_count_files)
        print("bbox数量:", annotation_count)
        print("类别数量:", category_count)

        counter = Counter()
        for ann in annotations:
            counter[ann["category_id"]] += 1

        print("类别分布:", counter)
        print("类别名称:")

        categories = []
        for c in data["categories"]:
            print(c["id"], c["name"])
            categories.append({"id": c["id"], "name": c["name"]})

        split_report["image_count_annotation"] = image_count_annotation
        split_report["image_count_files"] = image_count_files
        split_report["annotation_count"] = annotation_count
        split_report["category_count"] = category_count
        split_report["category_distribution"] = dict(counter)
        split_report["categories"] = categories
        split_report["missing_images"] = missing_images
        if missing_images:
            split_report["status"] = "failed"
        report["splits"][split] = split_report

    if any(split_data["missing_images"] for split_data in report["splits"].values()):
        report["status"] = "failed"

    return report


def main():
    check_raw_data(DEFAULT_ROOT)


if __name__ == "__main__":
    main()