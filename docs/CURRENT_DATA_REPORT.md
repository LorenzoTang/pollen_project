# 当前 COCO 数据验收报告

> 本文保留的是 2026-09-23 的旧数据验收记录。2026-10-12 默认训练集已更新为 107 张图片、787 个框、15 类；当前报告见 `TRAINING_READINESS_V2.md` 和 `DATA_V2_AUDIT.json`。

验收日期：2026-09-23

## 数据来源

- 外部只读来源：`C:\Users\唐\Desktop\pollen_project\data`
- 项目 raw 副本：`data/raw/标签数据(1)`
- canonical annotation：`coco/dataset.json`
- canonical images：`coco/images/*.jpg`
- `dataset.json` 与 `dataset_zh.json` 内容等价，只使用前者，避免重复计数。

## 原始数据验收

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

## 固定切分结果

配置：seed 42，train ratio 0.8。准备脚本先对文件名稳定排序，再使用独立随机数生成器打乱，并确保每个真实类别至少进入训练集。

- train：20 图，182 框，6 类均有标注
- valid：4 图，16 框
- train/valid 源文件名无重叠
- processed 类别 ID：连续 1–6
- 文件映射：`data/processed/rfdetr_pollen/split_manifest.json`
- 统计：`data/processed/rfdetr_pollen/split_summary.json`

验证集只含樟 9 框、楝 7 框。其余类别没有足够样本支持同时保持训练覆盖和独立验证覆盖；脚本不会通过复制图片制造数据泄漏。

## 结论

数据在工程层面已满足 RF-DETR bbox 训练输入要求。主要风险不是格式，而是样本规模和类别不均衡：黑松、白玉兰不能支撑可信泛化评估。当前训练适合作为可复现基线和后续数据扩充的起点，不应作为最终科研结果。