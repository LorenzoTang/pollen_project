# RFDETR_INFERENCE_INTERFACE.md

## 1. 目标

本文件用于确认当前项目中 RF-DETR 的 inference 和 checkpoint 使用方式，为后续 evaluation 实现做接口准备。

本次仅做代码/依赖分析，不实现 evaluation，不修改任何项目代码。

---

## 2. 已确认的 RF-DETR 版本

- 当前安装的 `rfdetr` 版本：`1.8.3`

---

## 3. 当前可用接口

通过仓库代码和临时接口探测，当前可确认：

- `RFDETRNano()`
- `RFDETRMedium()`
- `model.train(...)`
- `model.predict(...)`

未发现以下公开接口：

- `evaluate(...)`
- `validate(...)`
- `load(...)`
- `save(...)`
- `from_pretrained(...)`

---

## 4. checkpoint 使用方式

### 4.1 训练后的 checkpoint 加载方式

从当前版本文档与接口探测结果看：

- `model.train(...)` 是通过 PyTorch Lightning 训练栈完成的
- 文档说明训练完成后，底层 `nn.Module` 会同步回 `self.model.model`
- 这意味着训练完成后模型对象本身仍可继续直接调用 `predict(...)`

### 4.2 当前结论

目前没有发现显式的 `load()` / `save()` 方法。
因此，后续若要做 evaluation，需要进一步确认：

- 训练产物 `.pth` 的保存位置
- 是否能通过实例化模型后自动恢复 checkpoint
- 或者是否需要从 `output_dir` 下的 checkpoint 文件进行显式加载

### 4.3 风险说明

checkpoint 具体加载路径和恢复方式仍需要结合训练产物进一步确认，但至少可以确定：

- 训练后模型对象可继续用于 `predict(...)`
- 当前仓库尚未封装独立加载逻辑

---

## 5. `model.predict(...)` 返回结构

根据接口文档和临时探测结果：

- `predict(...)` 的返回类型是：
  - 单张图像：`supervision.Detections` 或 `supervision.KeyPoints`
  - 多张图像：对应对象列表

对于当前 bbox detection 场景，重点是 `supervision.Detections`。

### 5.1 `predict(...)` 关键签名

```python
predict(
    images,
    threshold=0.5,
    shape=None,
    patch_size=None,
    include_source_image=True,
    **kwargs
)
```

### 5.2 预测结果中可取字段

对于 `supervision.Detections`，可直接获取：

- `detections.xyxy`：bbox，格式为 `[x1, y1, x2, y2]`
- `detections.confidence`：置信度分数
- `detections.class_id`：类别 ID

文档还说明：

- `detections.metadata["source_image"]` 可获取原始图像（如果 `include_source_image=True`）

---

## 6. 如何获得 bbox / confidence / category id

对于 bbox detection 结果：

### bbox
- 来自 `detections.xyxy`
- 每个目标一个 `[x1, y1, x2, y2]`

### confidence score
- 来自 `detections.confidence`
- 与阈值 `threshold` 配合使用

### category id
- 来自 `detections.class_id`
- 需要注意训练后类别索引与 COCO `category_id` 的映射关系

---

## 7. 是否可以直接转换为 COCO detection result 格式

### 结论
**可以，原则上可直接转换。**

原因：

- COCO detection 结果需要的核心字段包括：
  - `image_id`
  - `category_id`
  - `bbox`
  - `score`

而 `supervision.Detections` 已经提供了：

- `xyxy`
- `confidence`
- `class_id`

因此只需要做格式转换：

1. 将 `xyxy` 转成 COCO 所需的 `[x, y, w, h]`
2. 将 `confidence` 映射为 `score`
3. 将 `class_id` 映射到 COCO `category_id`
4. 绑定对应的 `image_id`

### 注意事项

- 如果是 fine-tuned detection 模型，`class_id` 通常是 0-based 索引
- COCO annotation 中的 `category_id` 可能不是连续编号
- 因此需要显式建立类别映射表，避免评估指标偏差

---

## 8. 推荐的后续评估接入方式

基于当前接口，推荐后续 evaluation 按以下方式实现：

1. 训练后保留 checkpoint 与 config
2. 使用 RF-DETR 模型对象进行 `predict(...)`
3. 读取 `supervision.Detections`
4. 转成 COCO detection 结果列表
5. 再交给 `pycocotools.COCOeval`

---

## 9. 结论

当前版本 RF-DETR 的 inference 接口已经足够支持后续 bbox evaluation 的基础工作：

- 能预测 bbox
- 能输出 confidence
- 能输出 class_id
- 能按 COCO detection 格式进行结果转换

但 checkpoint 的显式加载方式仍需要进一步确认训练产物位置与恢复方式。
