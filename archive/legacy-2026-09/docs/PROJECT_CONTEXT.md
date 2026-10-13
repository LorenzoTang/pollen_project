# Pollen Detection Project Context

## Goal

Develop a computer vision system for pollen recognition and detection.

Main objectives:

1. Classify pollen categories.

2. Detect pollen objects in microscope images.

3. Build a research-quality detection pipeline.

## Current Model

Framework:

- RF-DETR

Deep learning:

- PyTorch

## Dataset Status

Current:

- Raw microscope images

- CSV annotation files

Processing target:

CSV annotations

        ↓

COCO format

        ↓

RF-DETR training dataset

## Project Structure

data/

    raw/

    processed/

src/

    datasets/

    detection/

    models/

    utils/

## Current Stage

Phase:

Dataset preprocessing

Current tasks:

- Understand annotation format

- Convert CSV to COCO

- Prepare RF-DETR training pipeline

## Experiment Principle

Every experiment should record:

- dataset version

- model version

- training parameters

- evaluation metrics