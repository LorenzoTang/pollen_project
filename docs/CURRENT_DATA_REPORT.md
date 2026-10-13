# 历史 COCO 数据验收报告 / Historical data acceptance report (2026-09-23)

[中文说明](#中文说明) · [English](#english) · [项目主页](../README.md#中文说明) · [Project README](../README.md#english)

## 中文说明

> 本文保留的是 2026-09-23 的旧数据验收记录，文件名保留以兼容已有引用。2026-10-12 默认训练集已更新为 107 张图片、787 个框、15 类；当前数据见[数据说明](../data/README.md#中文说明)和 [DATA_V2_AUDIT.json](DATA_V2_AUDIT.json)，首次试跑历史见 [TRAINING_READINESS_V2.md](TRAINING_READINESS_V2.md#中文说明)。下文路径、数据量与结论均对应当时状态。

验收日期：2026-09-23

### 数据来源

- 外部只读来源：`C:\Users\唐\Desktop\pollen_project\data`
- 项目 raw 副本：`data/raw/标签数据(1)`
- canonical annotation：`coco/dataset.json`
- canonical images：`coco/images/*.jpg`
- `dataset.json` 与 `dataset_zh.json` 内容等价，只使用前者，避免重复计数。

### 原始数据验收

- 图片：24
- 标注框：198
- 类别：6
- 所有 image/category ID 引用有效
- 所有 bbox 为四元素数值、宽高为正且未越界
- 24 张图片全部存在、可由 Pillow 解码，实际尺寸均与 COCO 记录一致
- COCO 内旧电脑绝对 `file_name` 已在 processed 输出中移除

| 原始 ID | 类别 | 标注框 |
|---:|---|---:|
| 0 | 夹竹桃 | 66 |
| 1 | 黑松 | 1 |
| 2 | 樟 | 75 |
| 3 | 构 | 5 |
| 4 | 楝 | 49 |
| 5 | 白玉兰 | 2 |

### 固定切分结果

配置：seed 42，train ratio 0.8。准备脚本先对文件名稳定排序，再使用独立随机数生成器打乱，并确保每个真实类别至少进入训练集。

- train：20 图，182 框，6 类均有标注
- valid：4 图，16 框
- train/valid 源文件名无重叠
- processed 类别 ID：连续 1–6
- 文件映射：`data/processed/rfdetr_pollen/split_manifest.json`
- 统计：`data/processed/rfdetr_pollen/split_summary.json`

验证集只含樟 9 框、楝 7 框。其余类别没有足够样本支持同时保持训练覆盖和独立验证覆盖；脚本不会通过复制图片制造数据泄漏。

### 结论

数据在工程层面已满足 RF-DETR bbox 训练输入要求。主要风险不是格式，而是样本规模和类别不均衡：黑松、白玉兰不能支撑可信泛化评估。当前训练适合作为可复现基线和后续数据扩充的起点，不应作为最终科研结果。

## English

> This is the historical acceptance record for the old dataset on 2026-09-23. The filename is retained for compatibility with existing references. On 2026-10-12, the default dataset changed to 107 images, 787 boxes, and 15 classes. See the [dataset guide](../data/README.md#english) and [DATA_V2_AUDIT.json](DATA_V2_AUDIT.json) for the current data, and [TRAINING_READINESS_V2.md](TRAINING_READINESS_V2.md#english) for the first smoke-test snapshot. Paths, counts, and conclusions below describe conditions at the time of this record.

Acceptance date: 2026-09-23.

### Data sources

- External read-only source: `C:\Users\唐\Desktop\pollen_project\data`
- Project raw copy: `data/raw/标签数据(1)`
- Canonical annotation: `coco/dataset.json`
- Canonical images: `coco/images/*.jpg`
- `dataset.json` and `dataset_zh.json` had equivalent contents; only the former was used to avoid double counting.

### Raw-data acceptance checks

- Images: 24
- Bounding boxes: 198
- Classes: 6
- All image/category ID references were valid.
- All bounding boxes had four numeric elements, positive width and height, and stayed within image bounds.
- All 24 images existed, could be decoded by Pillow, and had actual dimensions matching the COCO records.
- Absolute `file_name` paths from the old computer were removed in the processed output.

Chinese category names are retained to match the annotation labels.

| Original ID | Class | Boxes |
| ---: | --- | ---: |
| 0 | 夹竹桃 | 66 |
| 1 | 黑松 | 1 |
| 2 | 樟 | 75 |
| 3 | 构 | 5 |
| 4 | 楝 | 49 |
| 5 | 白玉兰 | 2 |

### Fixed split results

Configuration: seed 42, train ratio 0.8. Preparation sorted filenames deterministically, shuffled them with an independent random-number generator, and ensured every actual class appeared in training.

- Training: 20 images, 182 boxes, with annotations for all 6 classes
- Validation: 4 images, 16 boxes
- No overlap in source filenames between training and validation
- Processed category IDs: contiguous 1–6
- File mapping: `data/processed/rfdetr_pollen/split_manifest.json`
- Statistics: `data/processed/rfdetr_pollen/split_summary.json`

Validation contained only 9 boxes for 樟 and 7 for 楝. The remaining classes lacked enough examples to support both training and independent validation coverage; the script did not duplicate images to manufacture coverage and cause leakage.

### Conclusion at the time

The data met the engineering input requirements for RF-DETR bounding-box training. The main risks were sample size and class imbalance rather than format: 黑松 and 白玉兰 could not support reliable generalization evaluation. Training on this data was suitable as a reproducible baseline and a starting point for data expansion, not as a final research result.