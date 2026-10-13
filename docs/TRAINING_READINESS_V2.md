# 新数据与训练就绪检查 / Training-readiness snapshot (2026-10-12)

[中文说明](#中文说明) · [English](#english) · [项目主页](../README.md#中文说明) · [Project README](../README.md#english)

## 中文说明

> 历史快照：下文描述的是首次试跑时的状态。后续已实现顺序队列、失败恢复和标准 COCOeval，并完成 50 轮及 640 对照实验。当前用法见 [训练指南](TRAINING_GUIDE.md)，最新结果见 [分辨率实验](RESOLUTION_EXPERIMENT.md)；下文“尚未实现”等表述仅适用于当时。

> 产物状态更新（2026-10-13）：首次试跑的权重和检查点已清理，配置、日志与评估文件保留。下文权重大小记录的是当时产物，不能据此直接恢复或重评该试跑。

默认配置已切换到新数据，GPU 单轮训练已完成。可以运行单次 RF-DETR Nano 正式训练；尚未实现自动多实验队列，也未启动 50 轮或批量正式训练。

### 数据替换结果

- 来源：`data/数据/coco - 副本/dataset.json`，图片位于 `data/数据`。
- 两份 COCO `dataset.json`、`dataset_zh.json` 内容相同，仅使用前者。
- 159 张 JPG 中，107 张有这份 COCO 标注：787 个框、15 类。其余 52 张未作为负样本加入训练。
- 106 份 LabelMe 文件中，41 份指向当前目录缺失的图片，均无嵌入图片。未将其与现成 COCO 自动合并，以免重复或更改类别定义。
- 图片尺寸、文件存在性、ID 引用、框范围检查通过；107 张已标注图片没有 SHA-256 完全相同的副本。尚未检查近重复图片或采样批次关联。
- 新输出：`data/processed/rfdetr_pollen_v2`。旧原始数据、旧 processed 数据和历史模型保留。
- 随机种子 42，train_ratio=0.7；训练类别覆盖修正后，训练 76 张/599 个框，验证 31 张/188 个框。
- 原类别 ID 0–14 转为 1–15，处理后图片使用 split 内数字文件名；映射保存在 `split_manifest.json`。
- 原始交付的 269 个文件及 SHA-256、未纳入图片清单、缺失 LabelMe 图片清单见 `DATA_V2_AUDIT.json`。

训练集覆盖全部 15 类；验证集仅覆盖 9 类。夹竹桃、洋紫荆、楠、细叶合欢、广玉兰、土紫荆没有验证标注。广玉兰总共仅 1 个框，樟有 480 个框（约 61%）。当前验证结果不能代表全部类别，也没有独立测试集。

### 设备与实际验证

- RTX 5060，8151 MiB 显存，驱动 591.44；检查时桌面等任务占用约 3.4 GiB。
- 系统内存约 15.9 GiB；检查时 C 盘剩余约 47.7 GiB。
- Python 3.11.9，PyTorch 2.11.0+cu128，CUDA 12.8；训练实际使用 GPU 和 bf16 混合精度。
- Nano，batch_size=2，grad_accum_steps=1，num_workers=0，learning_rate=0.0001。
- 实验 `experiments/RF-DETR-Nano-v2-smoke-check`：2026-10-12 10:56:50 开始，10:57:26 完成，约 36 秒（包含构造模型），38 个训练 batch。
- 保存 `checkpoint_best_total.pth`、`checkpoint_best_ema.pth`、`last_ema.pth`、`last.ckpt`、`metrics.csv`、`training_config.json`；本次权重与检查点合计约 0.79 GiB。
- 训练器验证输出：mAP50≈0.1997，mAP50–95≈0.1467；仅验证流程可运行，不是最终模型结果。
- 11 项 unittest 通过，含新目录读取、Windows 原始绝对路径转换和防覆盖回归测试。
- TensorBoard 未安装，CSV 日志正常；Triton 不可用时匹配算法回退 CPU，未阻止训练。

### 代码检查与批量训练缺口

本次改动仅包括新数据目录兼容、baseline/smoke 配置切换、实验目录重名拒绝及回归测试。数据准备、训练预检、GPU 训练、权重保存均已实测。

现有 `src/run_pipeline.py` 只运行一个实验；没有批量任务清单、队列状态、失败继续、结果汇总。它每次尝试重新准备数据，不适合直接循环并覆盖共享数据目录。批量任务应复用一次生成的固定数据版本。

现有 `src/evaluate_rfdetr.py` 的 `evaluate_predictions` 将 precision×recall 记作 AP，并对 IoU 阈值平均，**不是标准 COCO mAP**。默认权重按修改时间选择，也不保证是最佳权重。独立评估仅用于加载和推理链路检查，其 mAP 字段不可用于正式实验排名。批量方案需统一采用训练器标准验证指标或 pycocotools COCOeval，并显式记录所选最佳权重。

恢复方面：SDK 支持 `training.resume`，完整状态应使用 `last.ckpt`。本次重名保护禁止写入已有实验目录；队列恢复需要设计受控恢复逻辑，不能直接用原实验名重跑。旧 6 类花粉模型不可直接用于新 15 类数据续训。本次使用 RF-DETR 通用预训练权重，SDK 自动重建 15 类检测头。

### 建议的首批任务与时间

确认后实现顺序队列：每次只运行一个 Nano 任务，固定本版数据划分，独立实验目录，记录配置/数据哈希、返回码、耗时和最佳指标；失败落盘，重启跳过已成功任务，恢复仅接受相同数据和类别的检查点。先提供 dry-run 再正式运行。

首批建议三组 50 epoch 实验：A 学习率 1e-4/累积1；B 学习率 5e-5/累积1；C 学习率 1e-4/累积2。均使用 batch_size=2、训练种子42，比较学习率和有效 batch size；不要并行争用这张 8GB GPU。

- 队列、规范评估、汇总和验证的实现工作量预计 1–2 小时；补充分批数据的固定划分继承/类别变更检查预计另 1–2 小时。这是工作量估计，尚未实施。
- 按本次 1 epoch 含启动约 36 秒估算，50 epoch 先预算每组 20–40 分钟，三组约 1–2 小时；单轮包含启动开销，不能精确外推，正式运行前几轮后再校准。
- 单组至少预留 1–2 GiB 输出空间，实际受 SDK 保存频率和训练长度影响。
- 后续新数据增多，耗时主要随训练/验证图片数、分辨率和每图目标数量变化，不能保持本次预估不变。

未来数据到达时先校验与去重，生成新版本，继承已有图片的划分，冻结评估样本；新类别需单独审查类别映射与评估覆盖。当前 splitter 在增加图片后重跑会改变旧图片划分，现阶段不能把这种重跑当作固定评估集的增量流程。

## English

> Historical snapshot: the following describes the state at the first smoke test. Sequential queues, failure recovery, and standard COCOeval were implemented afterward, and 50-epoch runs and the 640 comparison were completed. See the [training guide](TRAINING_GUIDE.md#english) for current usage and the [resolution experiment](RESOLUTION_EXPERIMENT.md#english) for the latest results. Statements such as “not yet implemented” below apply only to the date of this snapshot.

> Artifact update (2026-10-13): weights and checkpoints from the first smoke test have been removed; configuration, logs, and evaluation files remain. The weight size below describes the original artifacts. This smoke test can no longer be directly resumed or reevaluated from those removed weights.

The default configuration had switched to the new data, and one epoch of GPU training had completed. A single full RF-DETR Nano run was possible; an automatic multi-experiment queue was not yet implemented, and neither 50-epoch nor batch production runs had started.

### Dataset replacement results

- Source: `data/数据/coco - 副本/dataset.json`, with images in `data/数据`.
- The two COCO files, `dataset.json` and `dataset_zh.json`, had identical contents; only the former was used.
- Of 159 JPG images, 107 had annotations in this COCO file: 787 boxes and 15 classes. The other 52 were not added as negative training samples.
- Of 106 LabelMe files, 41 referenced images missing from the current directory, with no embedded image data. They were not automatically merged with the existing COCO file, to avoid duplicates or changes to class definitions.
- Image dimensions, file existence, ID references, and box bounds passed validation. No two of the 107 annotated images had identical SHA-256 hashes. Near duplicates and acquisition-batch relationships had not yet been checked.
- New output: `data/processed/rfdetr_pollen_v2`. Older raw data, processed data, and historical models were retained.
- Seed 42 and train_ratio=0.7; after ensuring training coverage of all classes, the split contained 76 training images/599 boxes and 31 validation images/188 boxes.
- Original category IDs 0–14 were converted to 1–15. Processed images used numeric filenames within each split; mappings were saved in `split_manifest.json`.
- The original delivery's 269 files and their SHA-256 hashes, excluded-image list, and missing LabelMe image list are recorded in [DATA_V2_AUDIT.json](DATA_V2_AUDIT.json).

Training covered all 15 classes, but validation covered only 9. 夹竹桃, 洋紫荆, 楠, 细叶合欢, 广玉兰, and 土紫荆 had no validation annotations. 广玉兰 had only one box in total; 樟 had 480 boxes (about 61%). These validation results could not represent all classes, and there was no independent test set. Chinese category names are retained to match the annotation labels.

### Hardware and actual checks

- RTX 5060, 8151 MiB VRAM, driver 591.44; desktop and other tasks occupied about 3.4 GiB at inspection time.
- About 15.9 GiB system memory; about 47.7 GiB free on drive C at inspection time.
- Python 3.11.9, PyTorch 2.11.0+cu128, CUDA 12.8; training actually used the GPU and bf16 mixed precision.
- Nano, batch_size=2, grad_accum_steps=1, num_workers=0, learning_rate=0.0001.
- Experiment `experiments/RF-DETR-Nano-v2-smoke-check` started at 10:56:50 on 2026-10-12 and finished at 10:57:26: about 36 seconds including model construction, with 38 training batches.
- It saved `checkpoint_best_total.pth`, `checkpoint_best_ema.pth`, `last_ema.pth`, `last.ckpt`, `metrics.csv`, and `training_config.json`. Weights and checkpoints originally totaled about 0.79 GiB.
- Trainer validation output: mAP50≈0.1997 and mAP50–95≈0.1467. These verified that the workflow ran; they were not final model results.
- Eleven unittest checks passed, including reading the new directory, converting original Windows absolute paths, and overwrite-protection regression tests.
- TensorBoard was not installed; CSV logging worked. When Triton was unavailable, matching fell back to the CPU without preventing training.

### Code review and batch-training gaps at that time

The changes at that point covered only compatibility with the new data directory, baseline/smoke configuration updates, rejection of duplicate experiment directory names, and regression tests. Data preparation, preflight, GPU training, and weight saving had all been exercised.

At the snapshot date, `src/run_pipeline.py` ran one experiment only. It had no batch manifest, queue status, continuation after failure, or result summary. It attempted to prepare data on each invocation, making it unsuitable for a loop that overwrote a shared data directory. Batch tasks needed to reuse one prepared, fixed data version.

The then-current `evaluate_predictions` in `src/evaluate_rfdetr.py` labeled precision×recall as AP and averaged it across IoU thresholds; this was **not standard COCO mAP**. Default weight selection used modification time, which did not guarantee the best checkpoint. Independent evaluation was suitable only for checking model loading and inference; its mAP field could not rank formal experiments. Batch evaluation needed consistent standard trainer validation metrics or pycocotools COCOeval, with explicit recording of the selected best checkpoint.

For recovery, the SDK supported `training.resume`; full-state recovery required `last.ckpt`. The duplicate-name protection prohibited writing into an existing experiment directory, so queue recovery required controlled logic rather than rerunning the original experiment name. The old six-class pollen model could not directly resume training on the new 15-class data. This smoke test used generic RF-DETR pretrained weights, with the SDK rebuilding the 15-class detection head automatically.

### Proposed initial tasks and time estimates at that time

The proposal was to implement a sequential queue after confirmation: run one Nano task at a time on the fixed dataset split, use separate experiment directories, and record configuration/data hashes, exit codes, timings, and best metrics. Persist failures, skip successful tasks on restart, and accept recovery checkpoints only for the same data and classes. Provide a dry run before execution.

The first proposed batch comprised three 50-epoch experiments: A with lr=1e-4/accum=1, B with lr=5e-5/accum=1, and C with lr=1e-4/accum=2. All would use batch_size=2 and training seed 42 to compare learning rate and effective batch size. They should not contend concurrently for the same 8GB GPU.

- Queue implementation, standard evaluation, summaries, and verification were estimated at 1–2 hours; preserving split assignments for incremental data and checking class changes at another 1–2 hours. These were work estimates for features not yet implemented at the snapshot date.
- Based on about 36 seconds for one epoch including startup, the initial budget was 20–40 minutes per 50-epoch run, or about 1–2 hours for all three. Startup overhead prevents precise extrapolation; recalibrate after the first few epochs of a full run.
- Reserve at least 1–2 GiB of output space per run; actual usage depends on SDK save frequency and training duration.
- With more data, runtime primarily varies with training/validation image count, resolution, and objects per image; these estimates cannot remain fixed.

For future data, the proposal was to validate and deduplicate first, create a new version, preserve existing image assignments, and freeze evaluation samples. New classes require a separate review of category mapping and evaluation coverage. Rerunning the splitter after adding images changes old assignments, so that rerun cannot be treated as an incremental workflow with a fixed evaluation set.