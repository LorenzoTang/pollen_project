from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .coco_validation import validate_coco_file
    from .config_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT, get_required_path, load_config
except ImportError:  # pragma: no cover
    from coco_validation import validate_coco_file
    from config_utils import DEFAULT_CONFIG_PATH, PROJECT_ROOT, get_required_path, load_config


def resolve_raw_layout(raw_root: Path) -> tuple[Path, Path]:
    raw_root = Path(raw_root)
    supplied_annotation = raw_root / "coco - 副本" / "dataset.json"
    if supplied_annotation.is_file() and not (raw_root / "coco" / "dataset.json").exists():
        return supplied_annotation, raw_root
    annotation_path = raw_root / "coco" / "dataset.json"
    image_candidates = (raw_root / "coco" / "images", raw_root / "label")
    image_root = next((path for path in image_candidates if path.is_dir()), None)
    if image_root is None:
        raise FileNotFoundError(f"Raw image directory not found. Tried: {image_candidates}")
    return annotation_path, image_root


def check_raw_data(raw_root: Path, *, decode_images: bool = True):
    annotation_path, image_root = resolve_raw_layout(raw_root)
    report = validate_coco_file(annotation_path, image_root, decode_images=decode_images)
    report.update(
        {
            "raw_root": str(Path(raw_root).resolve()),
            "annotation_file": str(annotation_path.resolve()),
            "canonical_image_root": str(image_root.resolve()),
        }
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate the canonical raw COCO pollen dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--no-decode", action="store_true", help="Skip Pillow image decoding checks.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    raw_root = get_required_path(config, "raw_data", "root", PROJECT_ROOT)
    report = check_raw_data(raw_root, decode_images=not args.no_decode)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())