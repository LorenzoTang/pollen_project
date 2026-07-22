# PROJECT_STATUS.md

## 项目背景

本项目是一个显微花粉图像目标检测项目，目标是使用深度学习模型识别不同类别的花粉。

当前主模型方向为 RF-DETR，现阶段主要使用 RF-DETR Nano 进行测试；后续在具备 GPU 环境后，可切换到 RF-DETR Medium 或更大模型。

---

## 当前数据阶段说明

当前仓库中的数据不是最终正式训练数据，而是用于开发和验证 pipeline 的测试数据。

当前数据特点：
- COCO detection 格式
- 包含 image、bbox、category
- 不包含 segmentation polygon

当前阶段只需要支持 bbox-based detection，不要引入 segmentation 逻辑。

未来正式训练数据预计会包含：
- 多张显微图像
- 多个花粉实例
- 多类别
- COCO 格式
- bbox + segmentation polygon

---

## 当前已有功能

仓库当前已经具备以下基础能力：

- YAML 配置驱动训练参数
- 模型类型配置化，支持 RFDETRNano / RFDETRMedium
- COCO bbox 数据检查脚本
- bbox 可视化脚本
- 基础训练入口脚本
- 基础数据目录与配置目录结构

---

## 已完成 Phase 1 内容

Phase 1 目前已经完成的内容包括：

- 训练入口支持从 YAML 配置读取参数
- 模型类型已配置化
- 支持未来模型切换：nano -> RFDETRNano，medium -> RFDETRMedium
- 已具备数据一致性检查能力
- 已具备 bbox 可视化能力
- 已有基础配置文件用于训练测试

当前判断：Phase 1 已完成基础 pipeline 骨架，但还没有形成完整的实验工程体系。

---

## 当前存在的问题

当前仓库还缺少以下工程化能力：

- 训练过程日志记录
- checkpoint 规范保存
- evaluation 流程
- inference 可视化流程
- 实验可复现记录
- 参数有效性校验
- 统一命令行入口
- 更完整的实验管理结构

另外，当前阶段不应修改数据流程以适配 segmentation，也不应大规模重构目录结构。

---

## 后续开发原则

后续开发需要遵守以下原则：

1. 不要大规模重构目录
2. 不要删除已有文件
3. 不要修改 COCO 数据格式
4. 不要加入 segmentation 模块
5. 每次只完成一个小任务
6. 修改前先说明计划
7. 修改后说明修改文件、修改原因和验证方式

---

## 下一阶段计划

下一阶段建议按以下顺序推进：

1. 稳定训练工程流程
   - 参数校验
   - 配置健壮性
   - 输出目录组织
   - 训练前基础检查

2. 完善 checkpoint 与实验记录
   - 保存模型权重
   - 保存配置快照
   - 记录训练参数与数据版本

3. 增加 evaluation 流程
   - validation 评估
   - 指标输出
   - 结果保存

4. 增加 inference 可视化
   - 推理结果绘制
   - 输出可视化图片

5. 增强可复现性
   - 固定随机种子
   - 记录完整配置
   - 保留实验上下文

---

## 当前结论

当前项目已经完成了基础 pipeline 骨架，适合继续向工程化训练流程推进。

下一步最优先的工作是：
- 稳定训练流程
- 规范 checkpoint 和日志
- 补齐 evaluation 与 inference 可视化
