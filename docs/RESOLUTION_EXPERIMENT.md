# Nano 640 分辨率对照实验

## 实验约束

- 固定使用已有 rfdetr_pollen_v2：76 张训练图片、31 张验证图片，不重新划分。
- 对照 A 组：50 轮、batch=2、grad_accum_steps=1、lr=0.0001、seed=42。
- 唯一显式新增参数为 `training.resolution: 640`，由训练入口透传给 SDK。
- SDK 自动同步位置编码尺寸（24 → 40）。保留 per-batch 多尺度和尺度扰动：A 的模型训练输入候选范围为 224–544，D 为 480–800，步长均为 32。日志中的单值 A=544、D=800 是数据预处理尺寸，之后模型仍按 batch 随机缩放。验证/推理尺寸为 A=384、D=640。
- 从同一 Nano 预训练权重开始；1 轮试跑仅用于验证训练和评估流程，不把试跑权重用于正式训练。
- 未更改编码器学习率、EMA、增强、采样或训练轮数。
- 本机 SDK 的 `from_checkpoint` 未从旧式 `args` 恢复分辨率（试跑权重实测回落到 384）；评估入口显式透传实验的 resolution 和关联位置编码尺寸，并将实际推理尺寸写入报告。A 组无 resolution 覆盖时行为保持不变，COCO 评估算法和阈值不变。最初的 1 轮试跑评估采用了默认 384，仅作为流程检查，不参与效果比较。

## 运行命令

```powershell
# 从仓库根目录执行
$root = (Get-Location).Path
& "$root\.venv\Scripts\python.exe" "$root\src\run_batch.py" --manifest "$root\configs\batch_resolution640_smoke.yaml"
& "$root\.venv\Scripts\python.exe" "$root\src\run_batch.py" --manifest "$root\configs\batch_resolution640.yaml"
```

队列入口会跳过已成功的任务。失败恢复需要显式提供 `--resume-failed`，不得改动同一队列的数据或配置。

## 对比和验收

两组均使用 `checkpoint_best_total.pth` 的独立 COCOeval 结果，不将训练器逐轮指标与独立指标混用。P/R/F1 工作点保持 confidence=0.5、IoU=0.5。

`scripts/compare_resolution.py` 接受 `--baseline`、`--candidate`、`--output` 三个路径，校验任务完成、50 轮、仅分辨率配置变化、数据哈希和评估协议一致后，输出中文 Markdown、JSON 和 CSV 对比，差值使用百分点。

本地最终比较目录：`experiments/resolution640-50epochs/comparison`（不随仓库分发）。

## 已完成结果（2026-10-13）

| 指标 | A：384 | D：640 |
| --- | ---: | ---: |
| mAP50–95 | 46.41% | 54.39% |
| mAP50 | 60.49% | 63.28% |
| mAP75 | 50.64% | 60.20% |
| Precision | 87.74% | 81.42% |
| Recall | 72.34% | 79.26% |
| F1 | 79.30% | 80.32% |
| TP / FP / FN | 136 / 19 / 52 | 149 / 34 / 39 |
| 最佳轮次（1 起算） | 40 | 36 |
| 训练耗时（分钟） | 19.76 | 25.25 |

### 有验证标注的类别

| 类别 | 验证框 | A AP50–95 | D AP50–95 | A Recall | D Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 白玉兰 | 3 | 1.99% | 5.72% | 0.00% | 0.00% |
| 山玉兰 | 4 | 22.13% | 27.47% | 0.00% | 0.00% |
| 楝 | 12 | 78.00% | 85.09% | 100.00% | 100.00% |
| 樟 | 131 | 67.04% | 75.10% | 85.50% | 93.89% |
| 山合欢 | 1 | 80.00% | 90.00% | 100.00% | 100.00% |
| 黑松 | 8 | 93.94% | 94.55% | 87.50% | 87.50% |
| 构 | 9 | 17.43% | 42.37% | 11.11% | 33.33% |
| 银杏 | 5 | 24.23% | 38.88% | 20.00% | 20.00% |
| 蓝花楹 | 15 | 32.93% | 30.32% | 13.33% | 13.33% |

夹竹桃、洋紫荆、楠、细叶合欢、广玉兰、土紫荆无验证标注，指标为 N/A。640 的总体 AP 和召回提升，但误检增加，不能据此认为弱类漏检已经解决。

结果来源为以下本地实验的 `results/training_report.json` 和独立评估报告：

- A：`experiments/batch-v2-50epochs-A-lr1e4-accum1-20261012_113149_215854`
- D：`experiments/resolution640-50epochs-D-nano640-lr1e4-accum1-20261013_104410_310958`

复核示例（需要本地保留上述产物）：

```powershell
$root = (Get-Location).Path
& "$root\.venv\Scripts\python.exe" "$root\scripts\compare_resolution.py" `
  --baseline "$root\experiments\batch-v2-50epochs-A-lr1e4-accum1-20261012_113149_215854" `
  --candidate "$root\experiments\resolution640-50epochs-D-nano640-lr1e4-accum1-20261013_104410_310958" `
  --output "$root\experiments\resolution640-50epochs\comparison"
```

## 结论限制

只有 9 类具有验证标注，没有独立测试集；一次 seed=42 的结果不能证明统计显著性或全部 15 类的泛化能力。GPU 耗时受运行负载影响，不能直接作为纯推理速度指标。