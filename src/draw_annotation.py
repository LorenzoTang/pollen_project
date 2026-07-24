# -*- coding: utf-8 -*-
"""用一张图片 + 其 COCO 标注 JSON 绘制带标注（分割掩膜/框/标签）的图片。"""
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import supervision as sv

# ============ 配置 ============
IMAGE_DIR = r"D:\Development\RF-DETR\__Data\训练数据\Train_Coco_Dataset_1\test" # 图片文件夹地址
ANN_JSON = os.path.join(IMAGE_DIR, "_annotations.coco.json")    # 标注文件名称
TARGET_IMAGE = "test_0001.jpg"          # 需绘制的图片的名称
OUTPUT = "annotation_result.jpg"       # 输出文件的名称
# =============================


def load_coco(ann_path):
    with open(ann_path, encoding="utf-8") as f:
        return json.load(f)


def build_detections(coco, image_name):
    # 找到目标图片的 image 记录
    img_rec = next((im for im in coco["images"] if im["file_name"] == image_name), None)
    if img_rec is None:
        raise ValueError(f"{image_name} 不在标注文件中")
    img_id = img_rec["id"]
    W, H = img_rec["width"], img_rec["height"]

    # 类别 id -> 名称
    id2name = {c["id"]: c["name"] for c in coco["categories"]}

    # 收集该图片的所有标注
    anns = [a for a in coco["annotations"] if a["image_id"] == img_id]

    boxes, class_ids, masks = [], [], []
    for a in anns:
        # COCO bbox 是 [x, y, w, h] -> 转成 [x1, y1, x2, y2]
        x, y, w, h = a["bbox"]
        boxes.append([x, y, x + w, y + h])
        class_ids.append(a["category_id"])

        # 分割掩膜（polygon 格式）
        seg = a.get("segmentation")
        if seg:
            mask = np.zeros((H, W), dtype=bool)
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                m = sv.polygon_to_mask(pts.astype(np.int32), resolution_wh=(W, H))
                mask |= m.astype(bool)
            masks.append(mask)

    detections = sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        mask=np.array(masks) if masks else None,
    )
    labels = [id2name[cid] for cid in class_ids]
    return detections, labels


if __name__ == "__main__":
    coco = load_coco(ANN_JSON)
    detections, labels = build_detections(coco, TARGET_IMAGE)

    image = np.array(Image.open(os.path.join(IMAGE_DIR, TARGET_IMAGE)).convert("RGB"))

    annotated = image.copy()
    if detections.mask is not None:
        annotated = sv.MaskAnnotator().annotate(scene=annotated, detections=detections)
    annotated = sv.BoxAnnotator().annotate(scene=annotated, detections=detections)
    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=labels)

    plt.figure(figsize=(12, 10))
    plt.imshow(annotated)
    plt.axis("off")
    plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"共 {len(detections)} 个标注，结果已保存到 {OUTPUT}")
