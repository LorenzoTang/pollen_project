# RFDETR_CHECKPOINT_RUNTIME_TEST.md

## 1. 目的

本次运行测试的目标是通过一次最小训练验证，确认 RF-DETR 1.8.3 实际生成的 checkpoint 文件结构。

本次仅进行运行验证与结果记录，不修改训练脚本，不实现 evaluation。

---

## 2. 本次运行结果

### 2.1 执行命令

运行了训练入口：

```bash
python src/train_rfdetr.py
```

### 2.2 训练启动阶段结果

训练流程进入了预检和模型初始化阶段，并创建了实验目录：

```text
experiments/RF-DETR-Nano_20260722_163713/
```

该目录下生成了：

- `config.yaml`
- `checkpoints/`
- `logs/`
- `results/`

### 2.3 运行失败原因

训练在真正开始前失败，原因是缺少训练依赖：

```text
ImportError: RF-DETR training dependencies are missing. Install them with `pip install "rfdetr[train,loggers]"` and try again.
```

更具体地说，缺少：

- `pytorch_lightning`

因此本次没有进入实际训练循环，无法生成 checkpoint 文件。

---

## 3. 实际生成文件列表

### 3.1 实验目录

实际生成的实验目录：

```text
experiments/RF-DETR-Nano_20260722_163713/
```

### 3.2 目录内容

目录中实际可见内容：

- `config.yaml`
- `checkpoints/`
- `logs/`
- `results/`

### 3.3 checkpoint 目录实际情况

通过目录检查确认：

- `experiments/RF-DETR-Nano_20260722_163713/checkpoints/`

当前为空，没有生成任何 checkpoint 文件。

---

## 4. checkpoint 结构结论

由于本次未能进入真实训练阶段，因此以下信息**无法从本次运行中确认**：

- checkpoint 文件扩展名
- checkpoint 文件大小
- checkpoint 顶层结构
- 是否存在可直接恢复模型权重的字段

### 4.1 可确认的结论

- 训练脚本确实会先创建 `experiments/<experiment_name>/checkpoints/`
- 但在训练依赖缺失时，该目录不会自动产生 checkpoint 文件

### 4.2 不能确认的结论

- 不能确认实际 checkpoint 文件名
- 不能确认 `torch.load()` 后的顶层 keys
- 不能确认权重字段是否为 `state_dict`、`model`、`weights` 或其他命名

---

## 5. 推荐加载方案

结合已有文档和本次运行结果，当前最稳妥的加载方案仍然是：

### 方案：显式指定 checkpoint 路径

evaluation 脚本应支持显式传入 checkpoint 文件路径，而不是依赖默认命名或自动猜测。

原因：

- 本次未确认默认命名规则
- 不同训练产物可能存在多个 checkpoint 文件
- 显式路径最利于复现与实验记录

---

## 6. 后续 evaluation 如何接入

当前 evaluation 仍建议按以下思路设计：

1. 从配置文件读取实验信息
2. 接收 `--checkpoint` 参数
3. 若未提供，则从实验目录中选择候选 checkpoint
4. 加载 checkpoint 后再执行 validation sweep

### 建议的输入约定

```text
--config configs/rfdetr_baseline.yaml
--checkpoint experiments/<experiment_name>/checkpoints/<file>.pth
--output-dir experiments/<experiment_name>/results
```

---

## 7. 后续验证步骤建议

要真正确认 checkpoint 文件结构，下一步需要：

1. 安装 RF-DETR 训练依赖
   - `pip install "rfdetr[train,loggers]"`
2. 再运行一次最小训练
3. 检查 `checkpoints/` 下实际生成文件
4. 对 checkpoint 执行 `torch.load(...)`
5. 记录顶层 keys 与权重字段

---

## 8. 总结

本次运行验证的结论是：

- 实验目录创建成功
- 训练在依赖缺失前就中断
- `checkpoints/` 目录为空
- 暂时无法确认实际 checkpoint 文件结构

因此，目前只能维持“显式 checkpoint 参数”的推荐方案，等待安装训练依赖后再进行下一轮真实训练验证。
