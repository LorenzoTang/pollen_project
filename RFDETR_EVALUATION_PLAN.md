# RFDETR_EVALUATION_PLAN.md

## 1. 当前可用评估接口

### 依赖版本
- 当前安装的 `rfdetr` 版本：`1.8.3`

### 仓库内已有接口使用情况
当前仓库中只发现以下 RF-DETR 相关调用：
- `RFDETRNano()`
- `RFDETRMedium()`
- `model.train(...)`
- `model.predict(...)`

仓库内**没有发现**现成的：
- `evaluate(...)`
- `validate(...)`
- `val(...)`
- `fit(...)`

### 现阶段结论
从仓库代码层面看，目前只确认了训练与推理接口，**没有直接可用的项目内 evaluation 封装**。

---

## 2. COCO bbox detection 的 validation evaluation 实现思路

当前数据是 COCO detection 格式，且只包含：
- image
- bbox
- category

不包含 segmentation polygon，因此评估也应按 **bbox detection** 路线实现。

### 推荐评估流程
1. 读取 validation 集 COCO JSON
2. 加载验证图片
3. 使用训练后的 RF-DETR 模型对每张图像做预测
4. 将预测结果转换为 COCO detection 格式
5. 使用 COCO API 计算指标

### 推荐评估工具
优先使用 `pycocotools`：
- `COCO`
- `COCOeval`

这与当前依赖中的 `pycocotools` 保持一致，也最适合 bbox 检测评估。

---

## 3. 可以获得的指标

使用 `pycocotools.COCOeval` 进行 bbox evaluation 时，通常可以得到：

- `mAP50-95`（主指标，COCO 标准）
- `mAP50`
- `precision`
- `recall`

### 说明
- `mAP50-95`：COCO 标准平均精度
- `mAP50`：IoU=0.50 下的平均精度
- `precision`：可从评估结果中提取或汇总
- `recall`：可从评估结果中提取或汇总

如果后续需要更细粒度指标，也可以补充：
- per-class AP
- per-class recall
- confusion-style analysis

---

## 4. 推荐实现方案

### 推荐方案：独立 evaluation 脚本
建议新增一个独立脚本，例如：
- `src/evaluate_rfdetr.py`

该脚本职责：
- 读取实验目录中的 `config.yaml`
- 加载 validation COCO 标注
- 加载训练后的模型 checkpoint
- 执行预测
- 将预测结果转成 COCO 结果格式
- 调用 `COCOeval` 输出指标

### 推荐输出
建议将结果写入实验目录：
- `experiments/<experiment_name>/results/eval_metrics.json`
- `experiments/<experiment_name>/results/eval_summary.txt`

### 推荐数据流
```text
validation images + val.json
        ↓
RF-DETR predict
        ↓
COCO result formatting
        ↓
pycocotools COCOeval
        ↓
metrics output
```

---

## 5. 需要新增哪些文件

建议最小新增文件如下：

1. `src/evaluate_rfdetr.py`
   - 主评估入口
   - 读取配置和 checkpoint
   - 运行 validation evaluation

2. `src/utils/coco_eval.py` 或 `src/utils/evaluation.py`
   - 可选
   - 用于封装 COCO 结果转换与指标提取

3. `experiments/<experiment_name>/results/`
   - 评估输出目录
   - 可由现有训练目录结构直接复用

### 最小化原则
如果要继续保持改动最小，可以先只新增：
- `src/evaluate_rfdetr.py`

其余工具函数后续再拆分。

---

## 6. 风险点

### 风险 1：RF-DETR 官方接口是否提供直接评估方法不明确
当前仓库没有发现 `evaluate(...)` 调用，因此不能假设官方提供了可直接使用的 evaluation API。

### 风险 2：预测结果到 COCO 格式的转换
需要确认 RF-DETR `predict(...)` 的返回结构，并把它稳定转换成 COCO detection 预测格式。

### 风险 3：类别 ID 映射
COCO 标注里的 `category_id` 与模型预测的类别索引可能需要显式映射，避免指标计算错误。

### 风险 4：验证集路径结构
评估脚本需要依赖稳定的验证集结构，建议延续当前：
- `dataset.path/annotations/val.json`
- `dataset.path/val/images`

### 风险 5：checkpoint 加载方式
需要确认训练后的权重保存位置和 RF-DETR 的加载接口是否一致，否则评估脚本无法直接复用训练产物。

---

## 7. 下一步建议

建议下一阶段按以下顺序推进：

1. 确认 RF-DETR checkpoint 保存/加载方式
2. 写独立 evaluation 脚本
3. 使用 `pycocotools` 做 bbox evaluation
4. 输出 mAP50、mAP50-95、precision、recall
5. 将评估结果保存到实验目录

---

## 8. 结论

当前项目适合采用 **“独立 evaluation 脚本 + pycocotools COCOeval”** 的实现方式。

这是最稳妥、最贴合当前 bbox-only 数据阶段的方案，也能避免引入不必要的复杂平台或 segmentation 逻辑。
