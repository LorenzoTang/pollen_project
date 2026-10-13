# RF-DETR 花粉检测训练指南

本指南对应当前仓库中的真实数据和配置，不适用于 README 中曾记录的旧 5 类 CSV 数据。

## 当前数据

- 原始 COCO：`data/数据/coco - 副本/dataset.json`
- 原始图片：`data/数据/`
- 当前纳入训练和验证的图片：107 张；另有 52 张未进入这份 COCO 标注
- 标注框：787 个
- 类别：15 类，完整清单与文件哈希见 `docs/DATA_V2_AUDIT.json`
- 固定随机种子：42
- 当前固定切分：train 76 张 / valid 31 张，599/188 个框；配置比例 0.7，保证训练类别覆盖后实际数量略有调整

原始 COCO 类别 ID 为 0–14。准备脚本会转换为 RF-DETR 使用的连续 ID 1–15，并将旧电脑绝对图片路径改为 split 内相对文件名。旧 processed 数据仍保留，新配置使用独立的 `rfdetr_pollen_v2`。最新实测和评估限制见 `docs/TRAINING_READINESS_V2.md`。

## 1. 创建环境

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_cuda.ps1
```

该脚本先从 PyTorch 官方 CUDA 12.8 索引安装 `torch/torchvision`，再安装 RF-DETR 训练依赖。RTX 50 系列不要直接依赖普通 PyPI 的 CPU Torch wheel。

训练前检查 CUDA：

```powershell
.\.venv\Scripts\python.exe src\check_environment.py --require-cuda
```

只有报告中 `status` 为 `passed` 且 `torch.cuda_available` 为 `true` 时，才建议开始正式训练。

## 2. 验证原始数据

```powershell
.\.venv\Scripts\python.exe src\check_data.py --config configs\rfdetr_baseline.yaml
```

该步骤检查 JSON 结构、ID 引用、bbox 正值及边界、文件存在性、图片可解码性和实际尺寸。

## 3. 生成训练数据

第一次生成：

```powershell
.\.venv\Scripts\python.exe src\prepare_rfdetr_dataset.py --config configs\rfdetr_baseline.yaml
```

明确替换已有 processed 数据：

```powershell
.\.venv\Scripts\python.exe src\prepare_rfdetr_dataset.py --config configs\rfdetr_baseline.yaml --overwrite
```

未提供 `--overwrite` 时，脚本不会删除已有数据。替换采用“临时目录生成并验证成功后再切换”的方式。

主要输出：

```text
data/processed/rfdetr_pollen_v2/
├── split_manifest.json
├── split_summary.json
├── train/
│   ├── 000001.jpg ...
│   └── _annotations.coco.json
└── valid/
    ├── 000001.jpg ...
    └── _annotations.coco.json
```

## 4. 训练前预检

```powershell
.\.venv\Scripts\python.exe src\train_rfdetr.py --config configs\rfdetr_baseline.yaml --preflight-only
```

预检不会构造模型或下载权重，也不会创建实验目录。

## 5. 开始正式训练

当前基线为 RF-DETR Nano、50 epochs、batch size 2：

```powershell
.\.venv\Scripts\python.exe src\train_rfdetr.py `
  --config configs\rfdetr_baseline.yaml `
  --experiment-name pollen_nano_seed42
```

也可以从 raw 检查、数据准备到训练一次执行：

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py `
  --config configs\rfdetr_baseline.yaml `
  --experiment-name pollen_nano_seed42 `
  --overwrite
```

训练产物位于：

```text
experiments/pollen_nano_seed42/
├── config.yaml
├── checkpoints/   # RF-DETR 权重与训练器产物
├── logs/train.log
└── results/
```

正式训练配置显式使用 `device: cuda` 和 `require_cuda: true`；CUDA 不可用时会在创建实验目录前失败，而不会退回 CPU 静默训练。

## 数据限制

当前数据为 15 类，训练集 76 张、验证集 31 张；验证集仅覆盖 9 类，樟约占全部标注的 61%，广玉兰仅 1 个框。没有独立测试集。验证集用于挑选最佳权重，不能将其结果视为无偏的最终泛化性能；无验证样本的类别 AP/recall/F1 显示 N/A。

## 6. 自动报告与标准评估

每次通过训练入口完成训练，都会加载 `checkpoint_best_total.pth` 进行独立评估并生成报告。`steps.evaluate` 的旧开关不再关闭这一步；报告失败时任务整体标为失败，权重仍保留。

实验目录的 `results` 中自动输出：

| 文件 | 内容 |
| --- | --- |
| `training_report.md` | 中文报告：计划/完成/最佳轮次、训练配置、数据规模、环境、损失、标准指标、限制 |
| `training_report.json` | 完整结构化报告，保留训练器全部逐轮指标 |
| `epoch_metrics.csv` | 每轮训练/验证总损失、分类/框回归/GIoU 及辅助损失、mAP、P/R/F1 等实际记录项 |
| `per_class_metrics.csv` | 每类 AP50、AP50–95、P/R/F1、TP/FP/FN 和有效标注数 |
| `training_curves.png` | 训练/验证损失和训练器验证指标曲线 |
| `evaluation_report.json` | 最佳权重的独立 COCOeval 指标、阈值、类别映射规则 |
| `predictions_coco.json` | 用于复核的 COCO 检测预测 |

AP 使用 COCOeval 的 IoU 0.50:0.05:0.95、101 点插值和 maxDets=[1,10,100]，输出 AP50、AP75、AP50–95、小/中/大目标 AP 及 AR。推理分数阈值为 0（保留模型输出候选，排除额外背景槽）。固定工作点的 P/R/F1 默认置信度 0.5、IoU 0.5、微平均，支持 `evaluation.confidence_threshold` 和 `evaluation.iou_threshold` 配置。

目标检测没有可枚举的真负例，因此不输出误导性的 accuracy。训练器的 F1 sweep 与独立评估的固定阈值 F1 口径不同，报告分开记录；用于跨实验比较的是独立评估指标。损失按训练器原始定义保存，不对缺失值补零。

`run_manifest.json` 记录配置、图片和标注文件的 SHA256、软件版本、源码哈希与实际传入的训练参数。`run_status.json` 记录训练/评估状态和本次调用耗时。恢复任务的耗时仅代表该次恢复调用，旧调用记录保留在原目录。

## 7. 固定数据的顺序批量训练

以下命令从仓库根目录执行；也可将 `$root` 指定为本机仓库的绝对路径：

```powershell
$root = (Get-Location).Path
& "$root\.venv\Scripts\python.exe" "$root\src\run_batch.py" --manifest "$root\configs\batch_v2.yaml" --dry-run
& "$root\.venv\Scripts\python.exe" "$root\src\run_batch.py" --manifest "$root\configs\batch_v2.yaml"
```

任务 A/B/C 均 50 轮、batch 2、seed 42，分别为 lr=1e-4/accum=1、lr=5e-5/accum=1、lr=1e-4/accum=2。队列复用已准备的数据，不调用重新划分脚本。每次训练独立目录，同一队列加进程锁；任务失败记录原因并继续下一任务。队列目录保存日志、状态以及跨实验 `summary.csv` / `summary.json`。再次执行会校验配置和数据并跳过成功任务。

失败后受控恢复：

```powershell
& "$root\.venv\Scripts\python.exe" "$root\src\run_batch.py" --manifest "$root\configs\batch_v2.yaml" --resume-failed
```

训练已完成时，仅重试评估和报告；训练未完成且存在 `last.ckpt` 时，使用完整状态恢复到新的尝试目录，保留旧日志、历史指标和最佳权重；没有恢复检查点时在新目录重新开始。配置或数据改变会拒绝恢复，需新建队列版本。

单任务也支持 `--resume-from`（原实验目录，配合新的 `--experiment-name`）与 `--report-only`（训练已完成的实验目录）。仅支持本版本产生 manifest 的实验；旧实验保留原样。队列汇总和状态可更新，历史训练目录不覆盖。

端到端短验证使用 `configs/batch_smoke.yaml`。通过流水线复用现有数据时，使用 `--reuse-prepared`；后续新增数据应建立新的 processed 版本。当前尚未实现新增图片自动继承旧 train/valid 划分。

## 8. 640 分辨率实验与手动重评

基础配置保留默认 384；`configs/batch_resolution640.yaml` 在相同基线上新增 `training.resolution: 640`。队列预检与运行：

```powershell
$root = (Get-Location).Path
& "$root\.venv\Scripts\python.exe" "$root\src\run_batch.py" --manifest "$root\configs\batch_resolution640.yaml" --dry-run
& "$root\.venv\Scripts\python.exe" "$root\src\run_batch.py" --manifest "$root\configs\batch_resolution640.yaml"
```

手动重评已有实验时必须使用该实验保存的完整配置，而非默认 baseline 配置或队列清单。以下为已完成 D 组的本地命令（新机器须替换为自己的实验目录）：

```powershell
$experiment = Join-Path $root 'experiments\resolution640-50epochs-D-nano640-lr1e4-accum1-20261013_104410_310958'
& "$root\.venv\Scripts\python.exe" "$root\src\evaluate_rfdetr.py" `
  --config "$experiment\config.yaml" --experiment-dir "$experiment"
```

该命令会更新实验中的评估文件。入口显式传入 resolution 和对应位置编码尺寸，并检查模型实际推理尺寸；应在 `evaluation_report.json` 确认 `inference_resolution` 为 640。详细结果见 [分辨率实验](RESOLUTION_EXPERIMENT.md)。