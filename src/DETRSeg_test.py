import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from rfdetr import RFDETRSegMedium
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 加载训练好的分割模型
    model = RFDETRSegMedium(
        num_classes=6,
        pretrain_weights=r"D:\Development\RF-DETR\Train\RFDETRSegMediumoutput\checkpoint_best_regular.pth"  # 修改为权重参数的地址
    )
    modelName = "RFDETRSegMedium"
    # 加载图片
    image = Image.open(r"D:\Development\RF-DETR\__Data\训练数据\Train_Coco_Dataset_2\test\test_0004.jpg")   # 修改为测试图片的地址

    # 推理
    results = model.predict(image)

    # 打印结果
    print(f"检测到 {len(results)} 个目标:\n")
    for i in range(len(results)):
        xyxy = results.xyxy[i]
        conf = results.confidence[i]
        class_name = results.data['class_name'][i]
        print(f"  [{i+1}] {class_name}: {conf:.3f}")
        print(f"       位置: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")

    # 可视化（用 supervision 的标注方式）
    import supervision as sv

    # 创建标注器：分割掩膜 + 边界框 + 标签
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # 准备标签
    labels = [
        f"{results.data['class_name'][i]} {results.confidence[i]:.2f}"
        for i in range(len(results))
    ]

    # 标注图片：先叠加分割掩膜，再画框和标签
    annotated_image = mask_annotator.annotate(
        scene=np.array(image),
        detections=results
    )
    annotated_image = box_annotator.annotate(
        scene=annotated_image,
        detections=results
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=results,
        labels=labels
    )

    # 保存和显示
    plt.figure(figsize=(12, 10))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.savefig(fr"{modelName}_output_result.jpg", bbox_inches='tight', dpi=150)    # 可指定输出图片的存放地址
    plt.show()
    print("\n结果已保存到 output_result.jpg")
