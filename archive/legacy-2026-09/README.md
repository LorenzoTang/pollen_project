# 历史文件归档 / Historical archive

归档日期：2026-10-13。此目录保存早期项目文档和独立试验脚本，原文件内容保持不变，原相对路径保留在本目录下。

这些文档描述的是历史状态，可能包含旧数据划分、旧电脑绝对路径和当时缺失的依赖；不代表当前环境或实验结果。当前说明见 [项目 README](../../README.md)、[数据说明](../../data/README.md)和[训练指南](../../docs/TRAINING_GUIDE.md)。

旧脚本仅供参考，不属于当前训练入口。`convert_coco.py` 会清空其配置的输出目录，`visualize.py` 使用旧机器路径，`nano-first-trial.py` 依赖早期 mini_dataset，`inference.py` 是通用 COCO 预训练模型演示。归档不保证这些脚本可以直接运行。

Archived on 2026-10-13. This directory preserves early project documents and standalone trial scripts without changing their contents. Their original relative paths are retained beneath this directory.

The documents describe historical conditions, including old dataset splits, machine-specific paths, and dependencies missing at that time. They do not describe the current environment or results. See the [project README](../../README.md), [dataset guide](../../data/README.md), and [training guide](../../docs/TRAINING_GUIDE.md) for current instructions.

The scripts are reference material, not current training entry points. `convert_coco.py` clears its configured output directory; `visualize.py` uses an old machine path; `nano-first-trial.py` requires the early mini_dataset; and `inference.py` demonstrates a generic COCO-pretrained model. The archived scripts are not guaranteed to run directly.

| 原路径 / Original path | 归档文件 / Archived file |
| --- | --- |
| PROJECT_STATUS.md | [PROJECT_STATUS.md](PROJECT_STATUS.md) |
| COCO_RFDETR_COMPATIBILITY_REPORT.md | [COCO_RFDETR_COMPATIBILITY_REPORT.md](COCO_RFDETR_COMPATIBILITY_REPORT.md) |
| PROCESSED_DATA_VALIDATION_REPORT.md | [PROCESSED_DATA_VALIDATION_REPORT.md](PROCESSED_DATA_VALIDATION_REPORT.md) |
| RAW_DATA_STRUCTURE_REPORT.md | [RAW_DATA_STRUCTURE_REPORT.md](RAW_DATA_STRUCTURE_REPORT.md) |
| docs/PROJECT_CONTEXT.md | [PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) |
| docs/ENVIRONMENT_DEPENDENCY_REPORT.md | [ENVIRONMENT_DEPENDENCY_REPORT.md](docs/ENVIRONMENT_DEPENDENCY_REPORT.md) |
| src/convert_coco.py | [convert_coco.py](src/convert_coco.py) |
| src/visualize.py | [visualize.py](src/visualize.py) |
| src/detection/nano-first-trial.py | [nano-first-trial.py](src/detection/nano-first-trial.py) |
| src/detection/inference.py | [inference.py](src/detection/inference.py) |