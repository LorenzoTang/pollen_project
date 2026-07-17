# Pollen Project

## Deep Learning-Based Microscopic Pollen Recognition with Self-supervised Feature Learning and Explainable AI

This project aims to develop an intelligent microscopic pollen recognition framework using deep learning techniques, focusing on automated pollen detection, feature representation learning, and explainable artificial intelligence (XAI).

The current development stage focuses on constructing a reliable pollen detection pipeline based on RF-DETR, including dataset processing, annotation conversion, and model training preparation. Future work will explore self-supervised feature learning and explainable AI methods for improving model robustness and interpretability.

---

# Overview

Microscopic pollen identification is an important task in environmental monitoring, allergy analysis, and biological research. Traditional pollen recognition methods mainly rely on expert observation, which can be time-consuming and require extensive domain knowledge.

This project investigates an automated pollen recognition system based on deep learning, aiming to achieve:

- Accurate pollen localization in microscopic images
- Robust feature representation learning
- Interpretable model decision analysis using explainable AI techniques

The overall framework:

```
Microscopic Images
        |
        v
Data Processing & Annotation Conversion
        |
        v
Object Detection Model (RF-DETR)
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
│   └── Saved model weights and training checkpoints
│
├── configs/
│   └── Training and experiment configuration files
│
├── data/
│   ├── raw/
│   │   └── Original microscopic pollen images and annotations
│   │
│   └── processed/
│       └── Processed datasets in COCO detection format
│
├── notebooks/
│   └── Data exploration, visualization, and experiment records
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

Each image may contain multiple pollen instances, where each annotation row represents one object bounding box.

Example:

```
Image A

Annotation 1
Annotation 2
Annotation 3
...
```

Therefore, the preprocessing pipeline groups annotations by image and converts them into a standard COCO object detection format.

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

Bounding Box Aggregation

          |
          v

COCO Format Generation

          |
          v

RF-DETR Training Dataset
```

The generated dataset has been verified through:

- Image and annotation consistency checking
- Bounding box visualization
- Category distribution analysis

---

# Current Dataset Statistics

The current baseline experiment uses five pollen categories.

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

The dataset has been converted into COCO detection format:

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

The current object detection model is based on RF-DETR, a transformer-based detection framework.

The goal is to detect and classify pollen instances from microscopic images.

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

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

## 1. Dataset Conversion

Convert raw CSV annotations into COCO format:

```bash
python src/convert_coco.py
```

---

## 2. Dataset Verification

Check image and annotation consistency:

```bash
python src/check_data.py
```

---

## 3. Bounding Box Visualization

Visualize generated annotations:

```bash
python src/visualize.py
```

---

## 4. Model Training

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
- [ ] Self-supervised feature learning for pollen representation
- [ ] Feature visualization and interpretation
- [ ] Explainable AI methods for model analysis
- [ ] Automated pollen recognition system deployment

---

# Experimental Goals

This project aims to investigate:

1. Whether transformer-based detection models can effectively recognize microscopic pollen structures.

2. How self-supervised feature learning can improve representation quality under limited labeled data conditions.

3. How explainable AI techniques can improve the interpretability of deep learning-based pollen recognition systems.

---

# Requirements

All required Python packages are listed in:

```
requirements.txt
```

---

# License

This project is developed for academic research purposes.