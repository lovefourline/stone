# RockClass V7 — 基于 ResNet50 的岩石薄片图像三分类

> 普通地质学与数智 AI 教学实践项目

---

## 📁 项目结构

```
stone_final/
├── README.md              ← 本文件
├── requirements.txt       ← Python 依赖
├── run.bat                ← Windows 一键训练
├── predict.bat            ← Windows 一键预测
├── train.py               ← 训练脚本
├── predict.py             ← 预测脚本
├── dataset/               ← 数据集（300 张图片）
│   ├── magmatite/         ← 岩浆岩 × 100
│   ├── metamorphic/       ← 变质岩 × 100
│   └── sedimentary/       ← 沉积岩 × 100
└── runs/                  ← 训练输出（自动生成）
    └── 20260609_183000/
        ├── best_model.pt         ← 最佳模型
        ├── report.csv            ← 训练报告
        ├── config.json           ← 超参数配置
        └── training_curves.png   ← 训练曲线图
```

---

## 🚀 快速开始

### 环境要求

| 项目 | 最低版本 |
|------|----------|
| Python | 3.10+ |
| GPU（可选） | NVIDIA + CUDA 12.x |

### 第一步：安装依赖

```bash
pip install -r requirements.txt
```

GPU 用户额外安装 CUDA 版 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 第二步：训练模型

**Windows 用户（推荐）**：双击 `run.bat`

**命令行**：

```bash
python train.py
```

训练完成后自动弹出图片选择窗口，可立即测试模型。

### 第三步：预测新图片

```bash
# 交互模式（弹窗选图）
python predict.py

# 预测单张图片
python predict.py my_rock.jpg

# 批量预测文件夹
python predict.py test_images/
```

---

## 🧠 模型架构

```
输入 (224×224×3)
    ↓
ResNet50 (ImageNet 预训练，主干冻结)
    ↓
Dropout(0.3) → FC(2048→256) → BN → ReLU → Dropout(0.3) → FC(256→3)
    ↓
输出: [岩浆岩概率, 变质岩概率, 沉积岩概率]
```

---

## 📊 训练策略

| 阶段 | Epoch | 学习率 | 可训练参数 | 说明 |
|:---:|:---:|:---:|:---:|---|
| S1 | 15 | 1e-2 | 分类头(~50万) | 仅训练分类头 |
| S2a | 10 | 1e-3 | layer4+分类头(~900万) | 解冻 ResNet 最深层 |
| S2b | 15 | 1e-4 | layer3+4+分类头(~1700万) | 渐进微调 |

**增强策略**：
- 优化器：SGD + Nesterov 动量 (momentum=0.9)
- 标签平滑：LabelSmoothing(ε=0.1)
- 参数平滑：EMA (decay=0.999)
- GPU 加速：混合精度训练 (AMP)
- 数据增强：翻转、旋转、裁剪、颜色抖动
- 测试增强：TTA（3 种变换取平均）
- 多轮验证：3 轮独立训练（不同随机种子），取最佳

---

## 📈 预期性能

| 指标 | 典型值 |
|------|--------|
| 验证准确率 | 95% ~ 100% |
| 训练时间 (GPU) | ~5 分钟/轮 (RTX 5060) |
| 训练时间 (CPU) | ~30 分钟/轮 |
| 模型大小 | ~95 MB |

---

## 📝 自定义训练

编辑 `train.py` 顶部的 `CONFIG` 字典：

```python
CONFIG = {
    'total_loops':  3,       # 训练轮数（越大越稳定）
    'epochs_stage1': 15,     # 分类头训练轮数
    'epochs_stage2a': 10,    # layer4 微调轮数
    'epochs_stage2b': 15,    # layer3+4 微调轮数
    'batch_size':    64,     # 批大小（显存不足可减小）
    'lr_stage1':     1e-2,   # 初始学习率
    'label_smoothing': 0.1,  # 标签平滑（0 = 关闭）
}
```

---

## 🔧 常见问题

**Q: 报错 `libiomp5md.dll already initialized`**
A: 已内置修复。如果仍出现，命令行加 `set KMP_DUPLICATE_LIB_OK=TRUE`。

**Q: 训练显示 `设备: cpu`**
A: 当前 Python 环境的 PyTorch 是 CPU 版，重装 GPU 版即可。

**Q: 显存不足**
A: 减小 `batch_size` 到 32 或 16。

---

## 📄 许可证

教育用途，自由使用。
