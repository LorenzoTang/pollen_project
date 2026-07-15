# pollen_project
基于自监督特征学习与可解释人工智能的显微花粉识别研究---Deep Learning-Based Microscopic Pollen Recognition with Self-supervised Feature Learning and Explainable AI，

GitHub仓库中的项目结构：
--README:也就是本文件，用于文本介绍项目概况
--checkpoint:用于保存训练好的模型
--configs:用于保存训练参数
--data:用于保存训练用的数据
--notebooks:用于数据探索；可视化；实验记录；日志————存放Jupyter Notebooks文件
--results：用于保存实验结果
--src:项目核心代码目录;包括数据加载，模型定义，训练；测试；推理
--requirements:记录项目运行所需要的Python库

### data/
Stores all datasets used in this project.

- `raw/`: Contains original microscopic pollen images without preprocessing.
- `processed/`: Contains processed datasets after preprocessing, including image resizing, normalization, augmentation, and dataset splitting.


### notebooks/
Contains Jupyter Notebook files for exploratory data analysis, visualization, and experimental studies.


### src/
Contains the core source code of the project.

- `dataset/`: Responsible for dataset loading, preprocessing, and PyTorch Dataset implementation.
- `models/`: Contains deep learning model architectures used for pollen classification.
- `utils/`: Contains auxiliary functions, such as visualization and evaluation tools.
- `train.py`: Main script for model training.
- `test.py`: Script for model evaluation.
- `inference.py`: Script for predicting pollen types from new images.


### configs/
Stores configuration files for experiments, including model settings, training parameters, and dataset paths.


### checkpoints/
Stores trained model weights and checkpoints generated during training.


### results/
Stores experimental outputs.

- `figures/`: Stores visualization results, such as loss curves, accuracy curves, and confusion matrices.
- `logs/`: Stores training logs and experiment records.


### requirements.txt
Contains all Python dependencies required to run this project.


### README.md
Provides an overview of the project, installation instructions, usage guidelines, and documentation.