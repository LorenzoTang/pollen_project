# Dataset / 数据集

[中文说明](#中文说明) · [English](#english) · [项目主页](../README.md#中文说明) · [Project README](../README.md#english)

## 中文说明

图片、原始标注和处理后数据均在本地管理，不上传 GitHub。请自行取得可使用的数据；本目录只提交说明文件和 `.gitkeep`。

### 当前数据（v2）

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

详见[训练指南](../docs/TRAINING_GUIDE.md#中文说明)、[数据审计](../docs/DATA_V2_AUDIT.json)与[训练就绪历史快照](../docs/TRAINING_READINESS_V2.md#中文说明)。旧 5/6 类数据和 CSV 转换流程不是当前 15 类实验的数据来源。

## English

Images, raw annotations, and processed datasets are managed locally and are not uploaded to GitHub. Obtain data you are permitted to use; only documentation and `.gitkeep` are tracked in this directory.

### Current dataset (v2)

There are 107 annotated images, 787 boxes, and 15 classes. With seed=42, the training/validation split contains 76/31 images and 599/188 boxes. Training covers all 15 classes, while validation covers only 9. There is no independent test set. Another 52 raw images are not included in this version of the COCO annotations and cannot simply be treated as negative samples without objects.

The default configuration, `configs/rfdetr_baseline.yaml`, uses this local layout:

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

Change `raw_data.root`, `processed_data.root`, and `dataset.path` in the configuration to use other locations. Preparation fixes old absolute image paths, maps raw category IDs 0–14 to 1–15, and saves both the mapping and fixed split. Existing output is not overwritten by default.

Run from the repository root:

```powershell
& .\.venv\Scripts\python.exe .\src\check_data.py --config .\configs\rfdetr_baseline.yaml
& .\.venv\Scripts\python.exe .\src\prepare_rfdetr_dataset.py --config .\configs\rfdetr_baseline.yaml
```

Controlled experiments reuse the same processed data; additions should produce a new dataset version. The current splitter does not automatically preserve existing assignments when adding data. Separately check images from the same slide, adjacent fields of view, and near duplicates to prevent training/validation leakage.

See the [training guide](../docs/TRAINING_GUIDE.md#english), [data audit](../docs/DATA_V2_AUDIT.json), and [historical training-readiness snapshot](../docs/TRAINING_READINESS_V2.md#english). The old 5/6-class datasets and CSV conversion workflow are not the data source for the current 15-class experiments.