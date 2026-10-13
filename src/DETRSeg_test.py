import argparse
import json
import os
from pathlib import Path

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSegMedium


DEFAULT_DATASET_DIR = Path(r"D:\Development\pollen_project_dev\COCO_Dataset_Raw2_Raw3")
DEFAULT_WEIGHTS = Path(
    r"D:\Development\pollen_project_dev\RFDETRSegMediumoutput-01\checkpoint_best_ema.pth"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model(weights: Path) -> RFDETRSegMedium:
    if not weights.is_file():
        raise FileNotFoundError(f"找不到模型权重: {weights}")
    return RFDETRSegMedium(num_classes=12, pretrain_weights=str(weights))


def evaluate_model(
    model: RFDETRSegMedium,
    dataset_dir: Path,
    split: str,
    eval_max_dets: int,
    output_dir: Path,
) -> None:
    metrics = model.evaluate(
        dataset_dir=str(dataset_dir),
        split=split,
        eval_max_dets=eval_max_dets,
        log_per_class_metrics=True,
    )

    print(f"\n{split} 集 COCO 评估指标:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_metrics.json"
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评估结果已保存: {output_path}")


def annotate_image(image: Image.Image, detections: sv.Detections) -> np.ndarray:
    scene = np.asarray(image.convert("RGB")).copy()
    if len(detections) == 0:
        return scene

    labels = [
        f"{detections.data['class_name'][i]} {detections.confidence[i]:.2f}"
        for i in range(len(detections))
    ]
    scene = sv.MaskAnnotator().annotate(scene=scene, detections=detections)
    scene = sv.BoxAnnotator().annotate(scene=scene, detections=detections)
    return sv.LabelAnnotator().annotate(scene=scene, detections=detections, labels=labels)


def predict_directory(
    model: RFDETRSegMedium,
    input_dir: Path,
    output_dir: Path,
    threshold: float,
    resolution: int,
    batch_size: int,
) -> None:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"找不到输入目录: {input_dir}")

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"目录中没有图片: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    print(f"开始批量推理: {len(image_paths)} 张图片")

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        predictions = model.predict(
            [str(path) for path in batch_paths],
            threshold=threshold,
            shape=(resolution, resolution),
            include_source_image=False,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]

        for image_path, detections in zip(batch_paths, predictions):
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            annotated = annotate_image(image, detections)
            result_path = output_dir / image_path.name
            Image.fromarray(annotated).save(result_path)

            objects = []
            for i in range(len(detections)):
                objects.append(
                    {
                        "class_name": str(detections.data["class_name"][i]),
                        "confidence": float(detections.confidence[i]),
                        "xyxy": [float(value) for value in detections.xyxy[i]],
                    }
                )
            records.append(
                {
                    "image": str(image_path),
                    "result": str(result_path),
                    "detections": objects,
                }
            )
            print(f"[{len(records)}/{len(image_paths)}] {image_path.name}: {len(objects)} 个目标")

    json_path = output_dir / "predictions.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n批量推理完成，结果目录: {output_dir}")
    print(f"预测明细已保存: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF-DETR 分割模型评估和批量推理")
    parser.add_argument("--mode", choices=("evaluate", "predict"), default="evaluate")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--eval-max-dets", type=int, default=500)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=432)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold 必须位于 0 到 1 之间")
    if args.batch_size < 1 or args.eval_max_dets < 1:
        raise ValueError("--batch-size 和 --eval-max-dets 必须为正整数")
    if args.resolution < 1:
        raise ValueError("--resolution 必须为正整数")

    model = build_model(args.weights)
    if args.mode == "evaluate":
        output_dir = args.output_dir or args.weights.parent
        evaluate_model(model, args.dataset_dir, args.split, args.eval_max_dets, output_dir)
        return

    input_dir = args.input_dir or (args.dataset_dir / args.split)
    output_dir = args.output_dir or (Path.cwd() / f"{args.split}_predictions")
    predict_directory(
        model,
        input_dir,
        output_dir,
        args.threshold,
        args.resolution,
        args.batch_size,
    )


if __name__ == "__main__":
    main()