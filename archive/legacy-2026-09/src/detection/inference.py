from pathlib import Path

import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

#这是第一个RF-DETR的实验性代码，用于尝试技术连能否打通。

ROOT = Path(__file__).resolve().parent
SKIP_IMAGES = {"cat_piano.jpg"}
OUTPUT_DIR = ROOT / "runs" / "rfdetr" / "pollen_try"
THRESHOLD = 0.25


def find_pollen_image() -> Path:
    """优先使用文件夹里的花粉显微图（排除示例图 cat_piano.jpg）。"""
    candidates = sorted(
        p for p in ROOT.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        and p.name not in SKIP_IMAGES
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 {ROOT} 中未找到花粉图像，请放入 .jpg/.png 等图片后再运行。"
        )
    return candidates[0]


def main() -> None:
    image_path = find_pollen_image()
    print(f"使用模型: RFDETRNano (COCO 预训练)")
    print(f"输入图像: {image_path.name}")
    print("说明: RF-DETR 预训练权重只能识别 COCO 常见物体，不能识别花粉种类。")
    print("      本次运行用于零成本试流程；要检测花粉需用标注数据微调模型。\n")

    model = RFDETRNano()
    image = Image.open(image_path).convert("RGB")

    detections = model.predict(image, threshold=THRESHOLD)

    print(f"检测框数量: {len(detections)}")
    if len(detections) == 0:
        print("未检测到任何目标（预训练模型通常无法识别花粉，属正常现象）。")
    else:
        for i, (class_id, confidence) in enumerate(
            zip(detections.class_id, detections.confidence), start=1
        ):
            label = COCO_CLASSES[int(class_id)]
            print(f"  [{i}] {label:<30} 置信度={confidence:.2f}")

    labels = [
        f"{COCO_CLASSES[int(class_id)]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    annotated_image = sv.BoxAnnotator().annotate(image.copy(), detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / f"{image_path.stem}_rfdetr.jpg"
    annotated_image.save(save_path)

    print(f"\n结果已保存到: {save_path}")
    annotated_image.show()


if __name__ == "__main__":
    main()
