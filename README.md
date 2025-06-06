# ResNet50+TinyViT_5M Knowledge Distillation

---

# 项目概述

## 本项目旨在将预训练的 **ResNet50** 作为教师模型，通过**知识蒸馏（Knowledge Distillation）**，将其“软知识”迁移到轻量级的 **TinyViT_5M** 学生模型。目标是在保持较高分类精度的同时，大幅减少参数量与计算开销。

- **教师模型**：ResNet50，50层深的残差网络，在图像分类任务中表现优异。
    
- **学生模型**：TinyViT_5M，拥有仅5M参数的小型视觉Transformer，经过快速蒸馏框架预训练以增强泛化能力。
    

> 参考文献：[Knowledge Distillation](https://arxiv.org/abs/1503.02531)、[ResNet](https://arxiv.org/abs/1512.03385)、[TinyViT](https://arxiv.org/abs/2207.10666)

---

# 方法

## 教师模型（ResNet50）

- 位于 `teacher.py`，加载 torchvision 官方预训练权重。
    
- 修改最后全连接层以适应**7类分类任务**。
    
- 自定义 `CustomDataset`，读取 `data/my_dataset/train` 和 `val`，并应用：
    
    - `Resize`、`RandomResizedCrop` 等增强。
        
- 训练细节：
    
    - 损失函数：**CrossEntropyLoss**
        
    - 优化器：**Adam**
        
    - 学习率策略：**StepLR**（每5个epoch下降10倍）
        
    - 训练最多20个epoch，**early stopping**，patience=3。
        
- 最优模型权重保存为 `88best_teacher_model.pth`。
    
- 生成全训练集对应的 logits 并保存为 `teacher_logits.pt`。
    

 [查看代码](https://github.com/AlexBybye/Resnet50-TinyViT_5M-KD/blob/master/teacher.py)

---

## 学生模型（TinyViT_5M）

- 位于 `tremendous_trial.py`，使用 timm 库创建 `tiny_vit_5m_224` 模型（drop_rate=0.1）。
    
- 封装 `Distiller` 类进行蒸馏训练：
    
    - **硬标签损失**：CrossEntropyLoss（label_smoothing=0.1）
        
    - **软标签损失**：KLDivLoss（reduction='batchmean'）
        
    - 蒸馏超参数：
        
        - 温度 `T=4.0`
            
        - 蒸馏权重系数 `α=0.7`
            
    - `index_mapping` 确保每个样本正确检索教师logits。
        
- 训练细节：
    
    - 优化器：**AdamW**
        
    - 学习率调度器：**CosineAnnealingLR**
        
    - batch_size=16，总训练20个epoch。
        

 [查看代码](https://github.com/AlexBybye/Resnet50-TinyViT_5M-KD/blob/master/tremendous_trial.py)

---

# 数据集

- 数据路径：`data/my_dataset`
    
- 结构：
    
    - `train/`、`val/` 下各有 **OK** 和 **NG（多类别）** 子目录。
        
    - 共7个类别。
        
- 分类逻辑参考 `classification.py`，使用 sklearn 分割数据集。
    

 [查看数据组织](https://github.com/AlexBybye/Resnet50-TinyViT_5M-KD/tree/master/data)

---

# 实验设置

## 超参数表

|超参数|值|
|:--|:--|
|教师学习率|5e-4|
|教师batch size|32|
|学生学习率|1e-4|
|学生batch size|16|
|训练轮次（epoch）|20|
|Optimizer|Adam / AdamW|
|学习率调度器|StepLR / Cosine|
|蒸馏温度（T）|4.0|
|蒸馏系数（α）|0.7|

## 训练流程

1. 运行 `python teacher.py` 训练教师模型并生成 `teacher_logits.pt`。
    
2. 将 `teacher_logits.pt` 和 `index_mapping` 放置在项目根目录。
    
3. 运行 `python tremendous_trial.py`，开始学生模型蒸馏训练。
    

---

# 实验结果

- 教师模型在验证集上达到理想准确率（通过控制台输出 `Val Acc: xx.xx%` 查看）。
    
- 学生模型通过蒸馏后，在验证集上表现稳定，收敛曲线平滑。
    
- **最终效果**：在大幅压缩参数规模的前提下，保持了优良的分类性能。
## 准确度
- 教师模型（teacher.py）:![{C8401259-0FF7-471A-A32E-E8D9911D77DC}](https://github.com/user-attachments/assets/a1760fa4-8dbb-4567-88c1-72eed52f1716)
## **（85%-92%）**
- 蒸馏模型（基于86%logits）:![12e558d0c26cdb25eb439e4c3f87522](https://github.com/user-attachments/assets/63b41bad-1b91-4eb9-b253-97c82d655f9f)
## **（79%-82%）**
---

# 结论

- 成功将 ResNet50 的深层知识（logits分布）蒸馏到 TinyViT_5M，训练出**轻量且高效**的学生模型。
    
- 自定义索引映射和数据加载机制，保证了蒸馏流程的**一致性和高效性**。
    
- 该流程在自定义小型分类任务上，兼顾了**精度、体积和推理速度**。
    

---

# 未来工作

- 尝试更多 Transformer 学生模型，如 MobileViT、Swin Transformer。
    
- 在大规模数据集（如 ImageNet-1k）上验证蒸馏性能。
    
- 结合剪枝（Pruning）与量化（Quantization）技术，进一步压缩模型体积。
    

---

# 使用说明

## 环境配置

```bash
pip install -r requirements.txt
```

---

# 参考文献

- He, K. et al. **Deep Residual Learning for Image Recognition**. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
    
- Wu, K. et al. **TinyViT: Fast Pretraining Distillation for Small Vision Transformers**. [arXiv:2207.10666](https://arxiv.org/abs/2207.10666)
    
- Hinton, G., Vinyals, O., & Dean, J. **Distilling the Knowledge in a Neural Network**. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
    
- AlexBybye. **Resnet50-TinyViT_5M-KD (GitHub)**. [项目链接](https://github.com/AlexBybye/Resnet50-TinyViT_5M-KD)
    

---
