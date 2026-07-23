# RAW_DATA_STRUCTURE_REPORT

## 数据目录结构

当前 `data/raw/` 下可见的原始数据结构为：

```text
data/raw/
└── 标签数据(1)/
    ├── coco/
    │   ├── dataset.json
    │   ├── dataset_zh.json
    │   └── output.xlsx
    └── label/
        ├── *.jpg
        └── *.json
```

### 结构特征
- `label/` 下是图片与同名 JSON 成对出现
- `coco/` 下存在导出的 COCO JSON 和 Excel 表
- 目录名称包含中文，后续脚本需要注意路径编码与转义

## 图片数据情况

### 已识别图片文件
- `label/` 目录下存在大量 `.jpg` 图片

### 图片数量
- 仅根据当前目录扫描结果，`label/` 下图片与 JSON 成对存在
- 由于当前环境未安装图像读取库，**本次无法直接统计像素级分辨率分布**

### 图片格式
- 已识别格式：`.jpg`

### 分辨率分布
- **待后续补充验证**
- 从 `coco/dataset.json` 中可见，至少部分图片记录的尺寸为：`2592 x 1944`
- 目前不能确认是否全部图片尺寸一致

## 标注文件情况

### 已识别标注文件类型
1. **LabelMe JSON**
   - 路径：`data/raw/标签数据(1)/label/*.json`
   - 结构特征：
     - 包含 `version`
     - 包含 `shapes`
     - `shapes` 内对象包含：`label`、`points`、`shape_type` 等字段
     - 示例中 `shape_type` 为 `polygon`

2. **COCO JSON**
   - 路径：`data/raw/标签数据(1)/coco/dataset.json`
   - 结构特征：
     - 包含 `images`
     - 包含 `annotations`
     - 记录了 `image_id`、`category_id`、`bbox`、`segmentation`、`iscrowd`、`area`
   - 说明：该文件不是纯 bbox-only 结构，而是带有 polygon segmentation 的 COCO 标注

3. **COCO 中文版 JSON**
   - 路径：`data/raw/标签数据(1)/coco/dataset_zh.json`
   - 推测用途：中文类别或中文字段映射版本

4. **Excel 文件**
   - 路径：`data/raw/标签数据(1)/coco/output.xlsx`
   - 推测用途：标注/转换过程中的中间导出表

### 标注内容抽样结论
- `label/*.json`：包含类别信息，且每个形状对应一个多边形区域
- `dataset.json`：包含 bbox、polygon segmentation、类别信息，以及 image-annotation 对应关系

## 当前识别出的数据格式

### 结论
- **原始标注来源：LabelMe 风格 JSON + JPG 图片配对文件**
- **已存在 COCO 格式导出文件**
- 当前数据不是单一格式，而是**LabelMe 原始标注 + COCO 导出混合结构**

### 格式判断
- **COCO JSON**：是
- **VOC XML**：未发现
- **YOLO txt**：未发现
- **CSV**：未发现
- **自定义格式**：存在辅助 Excel，但主标注不是自定义纯文本格式

## 是否可以直接转换为 COCO detection

### 结论
- **可以，且很可能已经具备 COCO 转换基础**

### 理由
- 原始 `label/*.json` 是 LabelMe polygon 标注
- 这类标注通常可稳定转换为 COCO detection 的 `bbox` + `category`
- `coco/dataset.json` 已表明项目已有 COCO 导出结果，说明转换链路大概率已经跑通

### 注意
- 当前 COCO 导出文件包含 `segmentation`，但你当前阶段目标是 **bbox-based detection pipeline**
- 因此后续工程应优先保证 bbox 训练流程，不必为当前阶段强依赖 segmentation

## 转换需求分析

### 当前阶段所需
- 读取 `label/*.jpg` 与同名 `label/*.json`
- 从 LabelMe `shapes` 中提取：
  - `label`
  - `points`
  - 基于 points 计算 bbox
- 输出符合 RF-DETR 训练所需的 **COCO detection bbox 格式**

### 未来可扩展
- 若正式训练数据包含 polygon segmentation：
  - 保留 `segmentation`
  - 同时生成 `bbox`
  - 兼容 detection 与 segmentation 两条数据流

### 当前不需要做的事情
- 不需要修改源码去适配 segmentation
- 不需要转换成 YOLO
- 不需要改动 `configs/`

## 下一步建议

1. 先对 `data/raw/标签数据(1)/label/` 做一次完整的图片尺寸与文件完整性验证
2. 确认 LabelMe JSON 中是否所有图片都具有对应标注
3. 明确类别集合与类别 ID 映射关系
4. 基于现有 LabelMe 原始标注，建立稳定的 COCO bbox 转换流程
5. 在转换前后增加数据一致性检查：
   - 图片是否存在
   - JSON 是否可解析
   - bbox 是否越界
   - 类别是否缺失
6. 继续保持当前阶段只做 bbox pipeline，不引入 segmentation 逻辑
