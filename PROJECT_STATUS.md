# PROJECT_STATUS.md

## Project Status

This document records the current state of the pollen detection project and is intended to be read-only status tracking.

---

## Phase 1: Pipeline Engineering and Baseline Infrastructure

### 已完成

- Raw dataset validation
- COCO annotation checking
- Dataset preparation pipeline
- COCO validation
- RF-DETR training integration
- Evaluation integration
- Unified pipeline entry: `src/run_pipeline.py`
- Pipeline report generation: `experiments/pipeline_report.json`

### 数据转换

- `prepare_rfdetr_dataset.py`
- 自动生成 `train/valid`
- 生成 RF-DETR 兼容 COCO 格式

### Pipeline

- `src/run_pipeline.py`

当前 pipeline 执行流程：

```text
data/raw
↓
run_pipeline.py
↓
raw data check
↓
dataset preparation
↓
COCO validation
↓
RF-DETR training
↓
evaluation
↓
experiment report
```

### 训练流程

- RF-DETR Nano baseline 已成功运行
- sanity check 已通过
- checkpoint 可以生成
- logging 已存在

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

**当前项目已进入 Phase 1：Pipeline Engineering and Baseline Infrastructure。**

当前已确认：

- raw 数据检查可以运行
- COCO JSON 可以正常加载
- train / valid 数据可以转换并验证
- pipeline 可以串联数据检查、数据转换和 COCO validation
- RF-DETR 训练入口仍然可单独运行
- checkpoint 已生成
- 评估流程已接入 pipeline 并可生成实验报告

---

## Current Constraints

- 当前数据集规模仍然是用于 pipeline 验证的测试数据
- Full-scale baseline experiment 尚未开始
- self-supervised learning 尚未开始
- explainable AI 模块尚未开始
- evaluation 系统已可用，但仍是基础版本
- 当前阶段只处理 bbox-based detection
- 不修改 COCO 数据格式
- 不引入 segmentation 逻辑
- 不进行大规模目录重构
- 不新增依赖

---

## 下一阶段规划

### Phase 2：Full baseline experiment

- Full dataset training
- Fixed train/validation/test split
- Reproducible configuration
- Complete metric reporting
- Model comparison experiments

### Phase 3

- self-supervised feature learning

### Phase 4

- explainable AI

---
