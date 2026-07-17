# Dataset / 数据集

[English](#english-version) | [中文](#中文版本)


---

# English Version


## Dataset Overview


This directory stores all datasets used in the pollen recognition project.


Due to dataset license restrictions, the original microscopic pollen images and annotations are not included in this repository.


Users need to prepare the dataset manually before running the preprocessing pipeline.


---

# Dataset Structure


The expected directory structure is:


```
data/

├── raw/
│
│   ├── Images/
│   │   └── Original microscopic pollen images
│   │
│   └── Annotations/
│       └── Original CSV annotation files
│
└── processed/
    
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

# Raw Dataset


The raw dataset contains:

- Microscopic pollen images
- CSV-based bounding box annotations


The annotation files contain:

- Pollen category
- Bounding box coordinates
- Image filename
- Image resolution


Example annotation format:


```csv
Class,x,y,w,h,file,resolution_x,resolution_y

1.Thymbra,1570,204,178,159,Lam.T.capX400wF(1029)(0)A.JPG,3072,2304
```


Each image may contain multiple pollen instances.

Therefore, multiple annotation rows may correspond to the same image.


Example:


```
Image_A.JPG

    |
    |-- pollen object 1
    |
    |-- pollen object 2
    |
    |-- pollen object 3
```


---

# Data Processing Pipeline


The raw dataset is converted into COCO object detection format.


Preprocessing script:


```
src/convert_coco.py
```


Pipeline:


```
Raw Dataset

(Image + CSV Annotations)

        |
        v

CSV Annotation Parsing

        |
        v

Grouping Annotations by Image

        |
        v

COCO Format Conversion

        |
        v

Processed Detection Dataset
```


---

# Dataset Verification


After conversion, the dataset can be verified using:


```bash
python src/check_data.py
```


Bounding boxes can be visualized using:


```bash
python src/visualize.py
```


Verification includes:

- Image and annotation consistency checking
- Bounding box statistics
- Category distribution analysis
- Annotation visualization


---

# Current Dataset Statistics


The current baseline experiment uses five pollen categories.


| Category | Images | Bounding Boxes |
|---|---:|---:|
| Thymbra | 12 | 157 |
| Erica | 8 | 186 |
| Castanea | 8 | 269 |
| Eucalyptus | 12 | 175 |
| Myrtus | 11 | 990 |


Total:

- Images: 51
- Annotated pollen instances: 1777


---

# Processed Dataset Format


The processed dataset follows the COCO object detection format.


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


Each annotation file contains:

- Image information
- Bounding boxes
- Category labels
- Dataset metadata


---

# Reproduction Steps


To reproduce the experiments:


1. Prepare the raw dataset.


2. Place images and annotations under:


```
data/raw/
```


3. Convert annotations:


```bash
python src/convert_coco.py
```


4. Verify dataset:


```bash
python src/check_data.py
```


5. Train model:


```bash
python src/train_rfdetr.py
```



---

<br>


# 中文版本


## 数据集简介


本目录用于存放花粉识别项目中使用的数据集。


由于数据集版权和许可限制，原始显微花粉图片以及标注文件不会上传至 GitHub。


使用者需要自行准备数据集后运行数据处理流程。


---

# 数据集结构


预期目录结构如下：


```
data/

├── raw/
│
│   ├── Images/
│   │   └── 原始显微花粉图片
│   │
│   └── Annotations/
│       └── 原始CSV标注文件
│
└── processed/
    
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

# 原始数据


原始数据包括：

- 显微花粉图片
- CSV格式目标框标注


标注文件包含：

- 花粉类别
- 边界框坐标
- 图片文件名
- 图片尺寸


示例：


```csv
Class,x,y,w,h,file,resolution_x,resolution_y

1.Thymbra,1570,204,178,159,Lam.T.capX400wF(1029)(0)A.JPG,3072,2304
```


由于一张图片可能包含多个花粉目标：

同一个图片文件名可能对应 CSV 中多行记录。


示例：

```
Image_A.JPG

    |
    |-- 花粉目标1
    |
    |-- 花粉目标2
    |
    |-- 花粉目标3
```


---

# 数据处理流程


原始CSV标注会被转换为 COCO 目标检测格式。


处理脚本：

```
src/convert_coco.py
```


流程：


```
原始数据

(图像 + CSV标注)

        |
        v

CSV解析

        |
        v

按照图片整合目标框

        |
        v

COCO格式转换

        |
        v

目标检测数据集
```


---

# 数据验证


转换完成后，可以使用：


```bash
python src/check_data.py
```


检查：

- 图片和标注对应关系
- Bounding Box数量
- 类别分布


可视化：


```bash
python src/visualize.py
```


---

# 当前数据统计


当前 baseline 实验使用5类花粉：


| 类别 | 图片数量 | 目标框数量 |
|---|---:|---:|
| Thymbra | 12 | 157 |
| Erica | 8 | 186 |
| Castanea | 8 | 269 |
| Eucalyptus | 12 | 175 |
| Myrtus | 11 | 990 |


总计：

- 图片数量：51
- 花粉实例数量：1777


---

# 处理后数据格式


处理后的数据采用 COCO Object Detection 格式。


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


标注文件包含：

- 图片信息
- 边界框
- 类别标签
- 数据集信息


---

# 复现实验步骤


1. 准备原始数据。


2. 将数据放入：


```
data/raw/
```


3. 执行数据转换：


```bash
python src/convert_coco.py
```


4. 检查数据：


```bash
python src/check_data.py
```


5. 开始训练：


```bash
python src/train_rfdetr.py
```