# RFDETR_CHECKPOINT_GUIDE.md

## 1. 目标

本文档用于确认 RF-DETR 1.8.3 的 checkpoint 保存与加载方式，为后续 evaluation 脚本接入 checkpoint 参数提供依据。

本次仅做代码/文档分析，不修改项目代码，不实现 evaluation。

---

## 2. 已确认信息

### 2.1 训练脚本中的 checkpoint 输出目录

根据 `src/train_rfdetr.py`：

- `model.train(...)` 传入的 `output_dir` 是：
  - `experiment_dir / "checkpoints"`
- 训练前会创建目录：
  - `experiments/<experiment_name>/checkpoints/`

同时还会创建：
- `experiments/<experiment_name>/logs/`
- `experiments/<experiment_name>/results/`

### 2.2 实验目录结构

训练脚本当前形成的结构是：

```text
experiments/<experiment_name>/
├── checkpoints/
├── logs/
├── results/
└── config.yaml
```

### 2.3 训练脚本对 checkpoint 的传参方式

训练调用为：

```python
model.train(
    dataset_dir=dataset_cfg["path"],
    epochs=training_cfg["epochs"],
    batch_size=training_cfg["batch_size"],
    grad_accum_steps=training_cfg["grad_accum_steps"],
    lr=training_cfg["learning_rate"],
    output_dir=str(experiment_dir / "checkpoints")
)
```

这说明当前项目已经把 checkpoint 输出位置统一到实验目录下的 `checkpoints/`。

---

## 3. RF-DETR 1.8.3 加载方式分析

### 3.1 公开接口情况

从接口探测结果看，`RFDETRNano` / `RFDETRMedium` 当前可见的公开方法中：

- `train(...)`
- `predict(...)`

没有发现公开的：

- `load(...)`
- `save(...)`
- `from_pretrained(...)`
- `evaluate(...)`

### 3.2 已确认的行为

`model.train(...)` 文档说明：

- 训练完成后，底层 `nn.Module` 会同步回 `self.model.model`
- 这意味着训练后的模型对象可以继续直接调用 `predict(...)`

### 3.3 仍未完全确认的点

当前尚未通过项目代码直接确认以下细节：

- checkpoint 文件是否会自动写入 `output_dir`
- checkpoint 文件的精确命名规则
- `RFDETRNano()` / `RFDETRMedium()` 是否支持通过构造参数直接加载本地 checkpoint
- 是否存在需要从训练目录中显式指定 checkpoint 文件路径的方式

---

## 4. checkpoint 保存形式

### 4.1 已确认

训练脚本将 `output_dir` 指向：

- `experiments/<experiment_name>/checkpoints/`

这表明 checkpoint 的保存会落在该目录中。

### 4.2 文件格式

根据 RF-DETR 的训练栈和项目当前使用方式，checkpoint 预计为：

- PyTorch checkpoint 文件
- 扩展名通常为 `.pth`

### 4.3 默认命名

当前仓库中**尚未直接确认**具体默认文件名。

因此现在只能确定：

- 目录：`experiments/<experiment_name>/checkpoints/`
- 格式：`.pth`（推定）
- 文件名：**待确认**

如果后续训练后生成了实际文件，建议再补充精确命名规则。

---

## 5. 新建模型后如何加载已有 checkpoint

### 5.1 当前可确认结论

目前没有发现官方公开的 `load()` 接口，也没有发现项目里已有的加载封装。

### 5.2 推荐判断方式

后续需要根据以下两类信息确认最终加载方案：

1. `rfdetr` 官方文档或源码
2. 训练后实际生成的 checkpoint 文件结构

### 5.3 可能的加载路径

基于目前信息，可能存在两类方式：

- **方式 A：模型实例化时自动恢复预训练/本地 checkpoint**
- **方式 B：通过训练后的 checkpoint 路径显式恢复模型状态**

当前还不能确定哪一种是 1.8.3 的标准方式。

---

## 6. evaluation 脚本如何接入 checkpoint 参数

### 6.1 推荐的脚本参数设计

后续 evaluation 脚本建议接收以下参数：

- `--config`：配置文件路径
- `--checkpoint`：checkpoint 文件路径
- `--output-dir`：结果输出目录

### 6.2 推荐接入方式

evaluation 脚本应优先支持：

1. 从 `config.yaml` 读取实验配置
2. 从命令行显式传入 checkpoint 路径
3. 若未传入，则默认从实验目录下推断 checkpoint 文件

### 6.3 推荐目录约定

建议约定：

```text
experiments/<experiment_name>/checkpoints/
```

evaluation 脚本应从这里读取训练产物，并将预测结果写入：

```text
experiments/<experiment_name>/results/
```

---

## 7. 推荐加载方案

在 checkpoint 机制未完全明确前，推荐采用以下优先级：

### 推荐方案 1：显式 checkpoint 路径

在 evaluation 脚本中增加 `--checkpoint` 参数，由用户明确传入 `.pth` 文件路径。

优点：
- 可控
- 易复现
- 不依赖默认文件名猜测

### 推荐方案 2：从实验目录自动查找最新 checkpoint

如果脚本执行环境已经固定实验目录，可以自动选择：

- `experiments/<experiment_name>/checkpoints/` 下最新的 `.pth`

优点：
- 使用方便

缺点：
- 依赖文件命名与时间排序
- 容易产生歧义

### 推荐结论

对于当前阶段，建议优先使用：

- **显式 checkpoint 路径**

这是最稳定、最适合实验管理和复现的方式。

---

## 8. 未确定风险

### 风险 1：默认 checkpoint 文件名未确认

当前还不知道 RF-DETR 1.8.3 在本项目训练流程中保存的默认文件名是什么。

### 风险 2：官方是否提供本地 checkpoint 加载接口未确认

目前没有发现公开 `load()` 方法，但仍需进一步确认源码或文档。

### 风险 3：checkpoint 与模型结构的兼容性

Nano / Medium 之间不能混用 checkpoint；模型结构参数必须匹配。

### 风险 4：evaluation 脚本需要知道具体 checkpoint 路径

如果不显式传参，脚本可能无法准确找到目标 checkpoint。

---

## 9. 结论

当前可以确定：

- 训练 checkpoint 会输出到 `experiments/<experiment_name>/checkpoints/`
- 训练脚本已经把输出目录标准化
- `train()` 结束后模型对象本身可继续用于 `predict()`

当前尚未完全确认：

- 默认 checkpoint 文件名
- 官方本地加载接口
- 是否需要构造函数直接传 checkpoint

### 最佳实践建议

后续 evaluation 脚本应支持显式 checkpoint 参数，避免依赖默认行为。
