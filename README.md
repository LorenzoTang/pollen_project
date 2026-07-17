# Pollen Project

## Towards Deep Learning-Based Microscopic Pollen Recognition with Self-supervised Feature Learning and Explainable AI

[English](#english-version) | [中文](#中文版本)


---

# English Version


## Overview

Microscopic pollen recognition is an important task in environmental monitoring, allergy analysis, and biological research. Traditional pollen identification methods mainly rely on expert observation, which is time-consuming and requires extensive domain knowledge.

This project aims to develop an intelligent microscopic pollen recognition framework based on deep learning techniques, focusing on:

- Automated pollen detection
- Robust feature representation learning
- Self-supervised feature learning under limited labeled data conditions
- Explainable Artificial Intelligence (XAI) for model interpretation


The current development stage focuses on building a reliable pollen detection pipeline using RF-DETR, including dataset processing, annotation conversion, and model training preparation.

Future work will investigate self-supervised learning strategies and explainable AI methods to improve model robustness and interpretability.


---

# Project Pipeline

The overall framework:

```
Microscopic Pollen Images

          |
          v

Dataset Processing & Annotation Conversion

          |
          v

COCO-format Detection Dataset

          |
          v

RF-DETR Object Detection Model

          |
          v

Pollen Detection Results

          |
          v

Feature Learning & Explainability Analysis
```


---

# Project Structure


```
pollen_project/

├── README.md
│
├── checkpoints/
│   └── Saved model weights and checkpoints
│
├── configs/
│   └── Training configuration files
│
├── data/
│   ├── raw/
│   │   └── Original microscopic pollen images and annotations
│   │
│   └── processed/
│       └── Processed datasets in COCO detection format
│
├── notebooks/
│   └── Dataset exploration, visualization and experiment records
│
├── results/
│   ├── figures/
│   │   └── Visualization results
│   │
│   └── logs/
│       └── Training logs and experiment records
│
├── src/
│   ├── convert_coco.py
│   ├── check_data.py
│   ├── visualize.py
│   └── train_rfdetr.py
│
└── requirements.txt
```


---

# Dataset Preparation


The original dataset contains:

- Microscopic pollen images
- CSV-based bounding box annotations


Each image may contain multiple pollen instances.

The annotation file stores one object per row:

```
Image A

Object 1
Object 2
Object 3
...
```


Therefore, the preprocessing pipeline groups annotations according to image names and converts them into COCO object detection format.


---

# Data Processing Pipeline


```
Raw Dataset

(Image + CSV Annotations)

          |
          v

CSV Annotation Parsing

          |
          v

Multi-object Bounding Box Aggregation

          |
          v

COCO Annotation Generation

          |
          v

RF-DETR Training Dataset
```


The processed dataset has been verified through:

- Image and annotation consistency checking
- Bounding box visualization
- Category distribution analysis


---

# Current Dataset Statistics


The current baseline experiment uses five pollen categories:


| Category | Images | Bounding Boxes |
|----------|-------:|---------------:|
| Thymbra | 12 | 157 |
| Erica | 8 | 186 |
| Castanea | 8 | 269 |
| Eucalyptus | 12 | 175 |
| Myrtus | 11 | 990 |


Total:

- Images: 51
- Annotated pollen instances: 1777


Processed dataset structure:

```
processed/

├── train/
│   └── images/
│
├── val/
│   └── images/
│
└── annotations/
    ├── train.json
    └── val.json
```


---

# Model


## RF-DETR


The current object detection framework is based on RF-DETR, a transformer-based detection model.

The objective is to automatically detect pollen instances from microscopic images.


Input:

```
Microscopic pollen image
```


Output:

```
Detected pollen objects

+

Bounding boxes

+

Pollen categories
```


---

# Installation


Install dependencies:

```bash
pip install -r requirements.txt
```


---

# Usage


## 1. Convert Dataset


Convert raw CSV annotations into COCO format:

```bash
python src/convert_coco.py
```


---

## 2. Verify Dataset


Check image and annotation consistency:

```bash
python src/check_data.py
```


---

## 3. Visualize Bounding Boxes


Visualize generated annotations:

```bash
python src/visualize.py
```


---

## 4. Train Model


Train RF-DETR:

```bash
python src/train_rfdetr.py
```


---

# Current Progress


## Completed

- [x] Project structure initialization
- [x] RF-DETR environment setup
- [x] Raw pollen dataset analysis
- [x] CSV annotation parsing
- [x] Multi-object annotation processing
- [x] COCO-format dataset conversion
- [x] Dataset consistency verification
- [x] Bounding box visualization pipeline


## In Progress

- [ ] RF-DETR baseline training
- [ ] Detection performance evaluation
- [ ] Model optimization


## Future Work

- [ ] Expand dataset to more pollen categories
- [ ] Self-supervised feature learning
- [ ] Feature representation analysis
- [ ] Explainable AI visualization
- [ ] Automated pollen recognition system deployment


---

# Research Goals


This project aims to investigate:


1. Whether transformer-based detection models can effectively recognize microscopic pollen structures.


2. How self-supervised learning can improve feature representation when labeled data is limited.


3. How explainable AI methods can improve the interpretability of deep learning-based pollen recognition systems.


---

# Requirements


All required Python packages are listed in:


```
requirements.txt
```



---

<br>


# 中文版本


## 项目简介


显微花粉识别是环境监测、过敏原分析以及生物研究中的重要任务。

传统花粉识别方法主要依赖专家人工观察，存在效率较低、依赖专业经验等问题。


本项目旨在构建一个基于深度学习的智能显微花粉识别框架，重点研究：

- 自动化花粉目标检测
- 鲁棒特征表示学习
- 有限标注数据条件下的自监督特征学习
- 基于可解释人工智能（XAI）的模型分析


目前项目主要围绕 RF-DETR 花粉检测系统展开，包括：

- 原始数据处理
- CSV标注解析
- COCO格式数据集构建
- 目标检测模型训练准备


后续将进一步探索自监督学习方法以及可解释人工智能技术。


---

# 项目流程


整体流程：

```
显微花粉图像

        ↓

数据处理与标注转换

        ↓

COCO目标检测数据集

        ↓

RF-DETR目标检测模型

        ↓

花粉检测结果

        ↓

特征学习与模型解释
```


---

# 数据处理


原始数据包括：

- 显微花粉图片
- CSV格式目标框标注


由于一张图片可能包含多个花粉目标，因此CSV文件中同一个图片名称可能对应多行标注。


数据处理流程：

```
原始数据

(图像 + CSV标注)

        ↓

CSV解析

        ↓

多目标标注整合

        ↓

COCO格式转换

        ↓

模型训练数据集
```


转换后的数据已经完成：

- 图片与标注一致性检查
- Bounding Box可视化验证
- 类别数量统计


---

# 当前数据规模


当前baseline实验选择5类花粉：


|类别|图片数量|目标框数量|
|-|-:|-:|
|Thymbra|12|157|
|Erica|8|186|
|Castanea|8|269|
|Eucalyptus|12|175|
|Myrtus|11|990|


总计：

- 图片数量：51
- 花粉实例数量：1777


---

# 当前进展


## 已完成

- [x] 项目结构搭建
- [x] RF-DETR环境配置
- [x] 原始花粉数据分析
- [x] CSV标注解析
- [x] 多目标标注处理
- [x] COCO格式数据转换
- [x] 数据一致性检查
- [x] Bounding Box可视化验证


## 当前进行

- [ ] RF-DETR baseline训练
- [ ] 检测性能评估
- [ ] 模型优化


## 后续计划

- [ ] 扩展更多花粉类别
- [ ] 引入自监督特征学习方法
- [ ] 进行特征表示分析
- [ ] 加入可解释人工智能模块
- [ ] 构建自动化花粉识别系统


---

# 项目目标


本项目希望探索：


1. Transformer检测模型在显微花粉识别任务中的有效性。


2. 自监督学习是否能够提升小规模标注数据条件下的特征表示能力。


3. 可解释人工智能方法是否能够提升深度学习模型的透明性和可信度。


---

# License


本项目用于学术研究目的。