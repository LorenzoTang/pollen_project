# PROCESSED_DATA_VALIDATION_REPORT

## 目标

验证 `src/prepare_rfdetr_dataset.py` 生成的 RF-DETR 训练前数据目录是否满足当前阶段的 bbox-based detection 需求。

---

## 数据生成结果

- 源数据：`data/raw/标签数据(1)/coco/dataset.json`
- 源图片目录：`data/raw/标签数据(1)/label/*.jpg`
- 输出目录：`data/processed/rfdetr_pollen/`
- 固定随机种子：`42`
- 划分比例：`train 80% / val 20%`

### 生成统计

- `train_images`: 19
- `val_images`: 5
- `train_annotations`: 116
- `val_annotations`: 82
- `categories`: 6

---

## 生成目录结构检查

输出目录已生成，结构如下：

```text
data/processed/rfdetr_pollen/
├── annotations/
│   ├── split_summary.json
│   ├── train.json
│   └── val.json
├── train/
│   └── images/
└── val/
    └── images/
```

目录结构符合当前任务要求。

---

## 图片检查结果

### train

- 图片数量：19
- 图片是否全部存在：是

### val

- 图片数量：5
- 图片是否全部存在：是

---

## COCO JSON 检查结果

### train.json

- images: 19
- annotations: 116
- categories: 6
- segmentation: 116

### val.json

- images: 5
- annotations: 82
- categories: 6
- segmentation: 82

### 说明

- annotation 中保留了 `bbox`、`category_id`、`image_id`
- `segmentation` 被保留，但当前流程不依赖它
- `area` 与 `iscrowd` 也被保留，便于 COCO 兼容性

---

## 一致性检查结果

### train

- annotation 的 `image_id` 是否都存在：是
- `category_id` 是否都存在于 categories：是
- `bbox` 是否为 `[x, y, width, height]`：是
- `bbox` 是否越界：否
- 图片文件是否全部存在：是

### val

- annotation 的 `image_id` 是否都存在：是
- `category_id` 是否都存在于 categories：是
- `bbox` 是否为 `[x, y, width, height]`：是
- `bbox` 是否越界：否
- 图片文件是否全部存在：是

---

## RF-DETR 训练可用性判断

### 结论

**可以作为 RF-DETR bbox-based detection 训练输入目录使用。**

### 原因

- 已按固定随机种子完成可复现划分
- 已生成 train / val 的图片目录和 COCO 标注文件
- 标注中保留了 bbox 和 category 信息
- 目录结构符合当前训练前处理需求
- 没有发现 image / annotation / category 不一致问题

### 备注

- 当前数据仍是开发与验证用测试数据，不代表最终正式训练集
- 当前阶段未引入 segmentation 训练逻辑
- 后续如果正式数据加入 polygon segmentation，可以在保持 bbox 逻辑不变的前提下再做扩展
