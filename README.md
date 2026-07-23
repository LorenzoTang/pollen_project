# PollenVision

- [English Version](#english-version)
- [中文版本](#中文版本)

---

<a name="english-version"></a>

# English Version

## Table of Contents

- [Overview](#overview)
- [Research Objectives](#research-objectives)
- [Current Status](#current-status)
- [Baseline Experiment](#baseline-experiment)
- [Project Structure](#project-structure)
- [Environment](#environment)
- [Dataset](#dataset)
- [Training](#training)
- [Technology Stack](#technology-stack)
- [Roadmap](#roadmap)

## Overview

PollenVision is a computer vision research project for microscopic pollen recognition.

Microscopic pollen recognition is a fine-grained visual understanding task. Different pollen categories can be highly similar in shape, texture, and size, which makes detection and classification challenging even for experienced human observers. The long-term goal of this project is to build a reliable and reproducible research pipeline for pollen object detection, feature learning, and model interpretation.

At the current stage, the project focuses on stabilizing a bbox-based detection pipeline using RF-DETR and preparing the engineering foundation for later research experiments.

## Research Objectives

The project is organized around three main research directions:

- Object detection for microscopic pollen instances
- Self-supervised feature learning for stronger representations under limited labels
- Explainable AI for interpreting model behavior and detection decisions

Relevant technical references:

- RF-DETR: https://github.com/roboflow/rf-detr
- DINOv2: https://github.com/facebookresearch/dinov2
- Captum: https://github.com/pytorch/captum

## Current Status

Phase 1 of the project is complete.

In this project, Phase 1 completion means:

- COCO data conversion pipeline is in place
- RF-DETR bbox detection pipeline is established
- Data loading validation has passed
- Smoke test training and validation can run successfully
- Checkpoints and logs can be generated

Based on the current repository state, the following parts of the RF-DETR baseline pipeline are already in place:

- COCO annotation conversion
- Dataset preparation
- RF-DETR training pipeline
- Validation pipeline
- Checkpoint generation
- Logging support in the training workflow

- The current repository contains a working smoke-test baseline built for pipeline validation rather than final scientific reporting.

This means the engineering pipeline is running end to end, but the current result does not represent final research performance.

## Baseline Experiment

The current baseline experiment is a smoke test using RF-DETR Nano.

Configuration summary from the repository:

- Configuration file: `configs/rfdetr_baseline.yaml`
- Task: bbox object detection
- Evaluation split: validation set
- Purpose: pipeline validation only

- Model: RF-DETR-Nano
- Dataset format: COCO
- Number of classes: 6
- Epochs: 1
- Batch size: 1
- Gradient accumulation steps: 1
- Learning rate: 0.0001

Dataset split used in the experiment:

- Train: 19 images
- Validation: 5 images

Observed metrics:

- mAP50-95: 0.2626
- Precision: 0.4352
- Recall: 0.4709
- F1: 0.4519
- EMA checkpoint mAP50-95: 0.2784

The current metrics below are for pipeline validation only:

- mAP50-95: 0.2626
- Precision: 0.4352
- Recall: 0.4709
- F1: 0.4519
- EMA checkpoint mAP50-95: 0.2784

These metrics should not be interpreted as final scientific results.

## Project Structure

The current repository structure is:

```text
pollen_project/
├── README.md
├── PROJECT_STATUS.md
├── configs/
│   └── rfdetr_baseline.yaml
├── data/
│   ├── .gitkeep
│   └── README.md
├── notebooks/
│   └── .gitkeep
├── results/
├── src/
│   ├── check_data.py
│   ├── convert_coco.py
│   ├── evaluate_rfdetr.py
│   ├── prepare_rfdetr_dataset.py
│   ├── train_rfdetr.py
│   └── visualize.py
├── requirements.txt
└── .gitignore
```

## Environment

The repository requirements are defined in `requirements.txt`.

Current dependencies include:

- torch
- torchvision
- rfdetr
- pandas
- numpy
- pillow
- tqdm
- pycocotools
- matplotlib
- opencv-python
- scikit-learn
- ipykernel
- jupyter

This project is intended to run in a local Python environment with PyTorch and RF-DETR support.

## Dataset

The dataset is not uploaded to GitHub.

Repository status and configuration indicate the following data policy:
- Current stage data is local development and validation data
- The project currently uses COCO bbox detection format
- The current stage is for pipeline testing only
- Future-stage data will be larger-scale formal research data
- Future-stage data may include segmentation polygons

Current stage data should not be confused with the final official training dataset.

Future stage data is expected to support formal scientific experiments.

## Quick Start

```bash
pip install -r requirements.txt
python src/prepare_rfdetr_dataset.py
python src/train_rfdetr.py
python src/evaluate_rfdetr.py
```

## Training

Dataset preparation:

```bash
python src/prepare_rfdetr_dataset.py
```

Training:

```bash
python src/train_rfdetr.py
```

Evaluation:

```bash
python src/evaluate_rfdetr.py
```

The repository also includes supporting scripts for data checking, COCO conversion, and visualization.

## Technology Stack

The current project stack includes:

- [PyTorch](https://pytorch.org/)
- [TorchVision](https://pytorch.org/vision/stable/)
- [RF-DETR](https://github.com/roboflow/rf-detr)
- [COCO API / pycocotools](https://github.com/cocodataset/cocoapi)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

## Roadmap

The development roadmap is organized in phases:

- Phase 2: Full baseline experiment
- Phase 3: Self-supervised feature learning
- Phase 4: Explainable pollen AI

## License

This project is released under the MIT License.

See the [LICENSE](LICENSE) file for details.

The source code is released under the MIT License.
Datasets are subject to their original licenses and are not included in this repository.

---

<a name="中文版本"></a>

# 中文版本

## 目录

- [项目简介](#项目简介)
- [研究目标](#研究目标)
- [当前状态](#当前状态-1)
- [基线实验](#基线实验)
- [项目结构](#项目结构-1)
- [环境配置](#环境配置)
- [数据集](#数据集)
- [训练方法](#训练方法)
- [技术栈](#技术栈)
- [未来规划](#未来规划)

## 项目简介

PollenVision 是一个面向显微花粉识别任务的计算机视觉研究项目。

显微花粉识别属于细粒度视觉理解问题。不同花粉类别在形态、纹理和尺度上往往非常接近，因此检测和分类难度较高，即使对经验丰富的人工观察者也是如此。本项目的长期目标，是构建一个可靠、可复现的花粉目标检测研究流程，并为后续的特征学习与模型解释研究提供稳定基础。

当前阶段，项目重点是使用 RF-DETR 稳定 bbox-based detection 流程，并完善后续科研实验所需的工程基础。

## 研究目标

本项目围绕三个主要研究方向展开：

- 显微花粉实例目标检测
- 在标注有限条件下进行自监督特征学习，以增强表征能力
- 通过可解释人工智能分析模型行为与检测决策

相关技术参考：

- RF-DETR: https://github.com/roboflow/rf-detr
- DINOv2: https://github.com/facebookresearch/dinov2
- Captum: https://github.com/pytorch/captum

## 当前状态

根据当前仓库状态，项目 Phase 1 已完成。

仓库中已经具备以下 RF-DETR 基线流水线能力：

- COCO 标注转换
- 数据集准备
- RF-DETR 训练流程
- 验证流程
- checkpoint 生成
- 训练流程中的日志记录

当前仓库中的结果属于用于流程验证的 smoke test，不应视为最终科研结论。

## 基线实验

当前基线实验是使用 RF-DETR Nano 进行的 smoke test。

从仓库配置中可以确认的实验设置如下：

- 模型：RF-DETR-Nano
- 数据格式：COCO
- 类别数：6
- 训练轮数：1
- batch size：1
- 梯度累积步数：1
- 学习率：0.0001

本次实验使用的数据划分为：

- 训练集：19 张图片
- 验证集：5 张图片

当前观察到的指标为：

- mAP50-95：0.2626
- Precision：0.4352
- Recall：0.4709
- F1：0.4519
- EMA checkpoint mAP50-95：0.2784

上述结果仅用于验证 pipeline 是否能够正常工作，不属于最终研究结果。

相关阶段状态记录在 `PROJECT_STATUS.md` 中。

## 项目结构

当前仓库结构如下：

```text
pollen_project/
├── README.md
├── PROJECT_STATUS.md
├── COCO_RFDETR_COMPATIBILITY_REPORT.md
├── PROCESSED_DATA_VALIDATION_REPORT.md
├── RAW_DATA_STRUCTURE_REPORT.md
├── configs/
│   └── rfdetr_baseline.yaml
├── data/
│   ├── .gitkeep
│   └── README.md
├── docs/
│   ├── ENVIRONMENT_DEPENDENCY_REPORT.md
│   └── PROJECT_CONTEXT.md
├── notebooks/
│   └── .gitkeep
├── results/
├── src/
│   ├── check_data.py
│   ├── convert_coco.py
│   ├── evaluate_rfdetr.py
│   ├── prepare_rfdetr_dataset.py
│   ├── train_rfdetr.py
│   └── visualize.py
├── requirements.txt
└── .gitignore
```

## 环境配置

项目依赖定义在 `requirements.txt` 中。

当前依赖主要包括：

- torch
- torchvision
- rfdetr
- pandas
- numpy
- pillow
- tqdm
- pycocotools
- matplotlib
- opencv-python
- scikit-learn
- ipykernel
- jupyter

本项目适合在本地 Python 环境中运行，并需要具备 PyTorch 与 RF-DETR 相关支持。

## 数据集

数据集不上传到 GitHub。

根据当前仓库状态与配置，数据管理方式如下：

- 原始数据由本地管理
- 处理后的数据由项目脚本自动生成
- 当前使用 COCO detection 格式
- 目前阶段只处理 bbox 标注
- 当前基线流程不包含 segmentation polygon

## 训练方法

数据集准备：

```bash
python src/prepare_rfdetr_dataset.py
```

训练：

```bash
python src/train_rfdetr.py
```

评估：

```bash
python src/evaluate_rfdetr.py
```

仓库中还提供了数据检查、COCO 转换和可视化相关脚本，便于完成完整的实验流程。

## 技术栈

当前项目使用的主要技术包括：

- [PyTorch](https://pytorch.org/)
- [TorchVision](https://pytorch.org/vision/stable/)
- [RF-DETR](https://github.com/roboflow/rf-detr)
- [COCO API / pycocotools](https://github.com/cocodataset/cocoapi)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

## 未来规划

项目后续规划分为以下阶段：

- Phase 2：完整基线实验
- Phase 3：自监督特征学习
- Phase 4：可解释花粉人工智能

## 快速开始

```bash
pip install -r requirements.txt
python src/prepare_rfdetr_dataset.py
python src/train_rfdetr.py
python src/evaluate_rfdetr.py
```

## 开源协议

本项目代码采用 MIT License 开源协议。

详细内容请查看 [LICENSE](LICENSE) 文件。

本项目代码遵循 MIT License。
数据集遵循其原始授权协议，本仓库不包含数据文件。