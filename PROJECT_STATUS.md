# PROJECT_STATUS.md

## Project Status

This document records the current state of the pollen detection project and is intended to be read-only status tracking.

---

## Phase 1: RF-DETR Baseline Pipeline

### 已完成

- 数据处理流程
- COCO 数据格式
- RF-DETR 训练链路
- checkpoint 保存
- logging

### 当前数据结构

当前 processed 数据集位于：

```text
data/processed/rfdetr_pollen/
├─ train/
│  ├─ 000001.jpg
│  ├─ 000002.jpg
│  ├─ ...
│  └─ _annotations.coco.json
└─ valid/
   ├─ 000001.jpg
   ├─ 000002.jpg
   ├─ ...
   └─ _annotations.coco.json
```

当前结构特点：

- `train/` 目录下图片直接位于 split 根目录
- `valid/` 目录下图片直接位于 split 根目录
- COCO JSON 中 `file_name` 保持纯文件名，例如 `000001.jpg`
- 当前数据为开发和验证 pipeline 的测试数据，不是最终正式训练数据
- 当前仅使用 COCO detection 的 bbox / category 信息，不涉及 segmentation polygon

### 最近一次成功实验

实验目录：

```text
experiments/RF-DETR-Nano_20260723_115102/
```

实验信息：

- 模型：RF-DETR Nano
- epoch: 1
- train images: 19
- valid images: 5

主要 metric：

- mAP 50:95: `0.2626`
- Precision: `0.4352`
- Recall: `0.4709`
- F1: `0.4519`
- EMA checkpoint mAP 50:95: `0.2784`

补充说明：

- smoke test 中先完成了 sanity check，再成功进入并完成 `Epoch 0`
- 训练流程已验证可以完整跑通 bbox detection pipeline

### checkpoint

已生成 checkpoint：

```text
experiments/RF-DETR-Nano_20260723_115102/checkpoints/checkpoint_best_regular.pth
```

同时训练日志中记录了 EMA 最终 checkpoint 的保存结果，说明实验目录内的 checkpoint 保存链路正常工作。

### 当前阶段结论

**Phase 1 已完成，bbox detection pipeline 已打通。**

当前已确认：

- COCO JSON 可以正常加载
- train / valid 图片路径正确
- DataLoader 正常工作
- PyTorch Lightning sanity check 通过
- 成功完成 1 epoch 训练
- checkpoint 已生成

---

## Current Constraints

- 当前阶段只处理 bbox-based detection
- 不修改 COCO 数据格式
- 不引入 segmentation 逻辑
- 不进行大规模目录重构
- 不新增依赖

---

## 下一阶段规划

### Phase 2

- 完整 20 类花粉数据训练
- baseline evaluation
- inference visualization

### Phase 3

- 模型改进研究
- self-supervised feature learning
- explainability

---

## Development Principles

1. 不要大规模重构目录
2. 不要删除已有文件
3. 不要修改 COCO 数据格式
4. 不要加入 segmentation 模块
5. 每次只完成一个小任务
6. 修改前先说明计划
7. 修改后说明修改文件、修改原因和验证方式
