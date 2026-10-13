# Dataset / 数据集

图片、原始标注和处理后数据均在本地管理，不上传 GitHub。请自行取得可使用的数据；本目录只提交说明文件和 `.gitkeep`。

## 当前数据（v2）

107 张已标注图片、787 个框、15 类。seed=42，训练/验证为 76/31 张，599/188 个框。训练集覆盖 15 类，验证集仅覆盖 9 类，没有独立测试集。另有 52 张原始图片未进入本版 COCO 标注，不能直接当作无目标负样本。

默认配置 `configs/rfdetr_baseline.yaml` 使用以下本地布局：

```text
data/
  数据/
    *.jpg
    coco - 副本/
      dataset.json
  processed/
    rfdetr_pollen_v2/
      split_manifest.json
      split_summary.json
      train/
        *.jpg
        _annotations.coco.json
      valid/
        *.jpg
        _annotations.coco.json
```

可修改配置的 `raw_data.root`、`processed_data.root` 和 `dataset.path` 以适配其他数据位置。准备脚本修正旧绝对图片路径，将原始类别 ID 0–14 映射到 1–15，并保存映射和固定划分。已有输出默认拒绝覆盖。

从仓库根目录执行：

```powershell
& .\.venv\Scripts\python.exe .\src\check_data.py --config .\configs\rfdetr_baseline.yaml
& .\.venv\Scripts\python.exe .\src\prepare_rfdetr_dataset.py --config .\configs\rfdetr_baseline.yaml
```

对照实验复用同一 processed 数据；数据增补应建立新版本。当前切分器尚不支持新增数据自动继承旧划分。需另行检查同一玻片、相邻视野和近重复图片，避免训练/验证泄漏。

详见[训练指南](../docs/TRAINING_GUIDE.md)与[历史数据审计](../docs/TRAINING_READINESS_V2.md)。旧 5/6 类数据和 CSV 转换流程不是当前 15 类实验的数据来源。

## English

Images, annotations, and generated datasets are not included in Git. Supply your own permitted COCO dataset and follow the layout above, or change the paths in `configs/rfdetr_baseline.yaml`.

The current v2 split contains 76 training images (599 boxes) and 31 validation images (188 boxes), with 15 classes overall and only 9 represented in validation. No independent test set is available. Preparation remaps category IDs to 1–15 and writes a split manifest. Reuse the prepared split for controlled experiments; adding data currently does not preserve existing split assignments automatically.