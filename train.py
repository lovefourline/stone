# -*- coding: utf-8 -*-
"""
RockClass — 基于 ResNet50 的岩石薄片图像三分类器 (V7 Final)
================================================================
策略: ResNet50(ImageNet预训练) + SGD/Nesterov + 三阶段渐进解冻
增强: LabelSmoothing + EMA + 混合精度 + TTA 预测
使用: python train.py    （训练完成后自动弹出选图窗口测试）
"""

# %%
# 解决 Anaconda + PyTorch 的 OpenMP DLL 冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- 标准库 ---
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# --- 深度学习 ---
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# --- 可选依赖 ---
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ╔══════════════════════════════════════════════════════════════╗
# ║                    设 备 检 测                               ║
# ╚══════════════════════════════════════════════════════════════╝

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ENABLED = (DEVICE.type == 'cuda')

print(f"\n  [设备] {DEVICE}")
if DEVICE.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  [GPU ] {gpu_name}  ({vram_gb:.1f} GB)")
    torch.backends.cudnn.benchmark = True
else:
    print("  [提示] 未检测到 GPU，将使用 CPU 训练（速度较慢）")

# ╔══════════════════════════════════════════════════════════════╗
# ║                   配 置 中 心                                 ║
# ╚══════════════════════════════════════════════════════════════╝

CONFIG = {
    # --- 路径 ---
    'data_dir':           Path(__file__).resolve().parent / 'dataset',
    'output_root':        Path(__file__).resolve().parent / 'runs',

    # --- 图像 ---
    'img_size':           224,
    'batch_size':         64,
    'num_workers':        0,

    # --- 训练策略 ---
    'total_loops':        3,        # 独立重复训练轮数（不同随机种子）
    'epochs_stage1':      15,       # 阶段1: 仅训练分类头
    'epochs_stage2a':     10,       # 阶段2a: 解冻 layer4
    'epochs_stage2b':     15,       # 阶段2b: 解冻 layer3+4

    # --- 学习率 ---
    'lr_stage1':          1e-2,
    'lr_stage2a':         1e-3,
    'lr_stage2b':         1e-4,

    # --- 优化器 ---
    'momentum':           0.9,
    'weight_decay':       1e-4,
    'nesterov':           True,

    # --- EMA ---
    'ema_decay':          0.999,

    # --- 损失函数 ---
    'label_smoothing':    0.1,

    # --- 验证集比例 ---
    'val_split':          0.2,
}

SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = CONFIG['data_dir']


# ╔══════════════════════════════════════════════════════════════╗
# ║                 输 出 目 录                                   ║
# ╚══════════════════════════════════════════════════════════════╝

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR   = CONFIG['output_root'] / TIMESTAMP
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH  = RUN_DIR / 'best_model.pt'
REPORT_PATH = RUN_DIR / 'report.csv'
CONFIG_PATH = RUN_DIR / 'config.json'
CURVE_PATH  = RUN_DIR / 'training_curves.png'

print(f"  [输出] {RUN_DIR}")

# --- 保存配置 ---
full_config = {
    **CONFIG,
    'timestamp':   TIMESTAMP,
    'device':      str(DEVICE),
    'amp_enabled': AMP_ENABLED,
}
if DEVICE.type == 'cuda':
    full_config['gpu_name'] = gpu_name
    full_config['vram_gb']  = round(vram_gb, 1)

# ╔══════════════════════════════════════════════════════════════╗
# ║                数 据 增 强                                    ║
# ╚══════════════════════════════════════════════════════════════╝

# ImageNet 标准均值/标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _base_tf(size=224):
    return transforms.Resize((size, size))

train_transform = transforms.Compose([
    _base_tf(CONFIG['img_size']),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.RandomResizedCrop(CONFIG['img_size'], scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    _base_tf(CONFIG['img_size']),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# TTA 变换：原图 + 水平翻转 + 垂直翻转
TTA_TRANSFORMS = [
    val_transform,
    transforms.Compose([
        _base_tf(CONFIG['img_size']),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        _base_tf(CONFIG['img_size']),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
]


# ╔══════════════════════════════════════════════════════════════╗
# ║              EMA（指数移动平均）                                ║
# ╚══════════════════════════════════════════════════════════════╝

class EMA:
    """对模型参数做指数移动平均，提升泛化能力。"""

    def __init__(self, model, decay):
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self):
        """将 EMA 参数替换到模型中（预测/保存前调用）"""
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data = self.shadow[name]

    def restore(self):
        """恢复原始参数"""
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[name]


# ╔══════════════════════════════════════════════════════════════╗
# ║              模 型 构 建                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def build_model(num_classes: int, verbose: bool = True) -> nn.Module:
    """
    构建 ResNet50 → 自定义分类头。
    主干全部冻结；分类头结构: Drop→FC256→BN→ReLU→Drop→FC(num_classes)
    """
    if verbose:
        print("    [1/3] 加载 ImageNet 预训练 ResNet50 ...")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    if verbose:
        print("    [2/3] 冻结主干网络 ...")
    for param in model.parameters():
        param.requires_grad = False

    if verbose:
        print("    [3/3] 构建自定义分类头 ...")
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║           训 练 / 验 证 / 预 测                                ║
# ╚══════════════════════════════════════════════════════════════╝

def train_one_epoch(model, loader, criterion, optimizer,
                    scaler=None, ema=None, pbar=None):
    """训练一个 epoch，返回 (avg_loss, accuracy)"""
    model.train()
    total_loss, correct, count = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct   += (outputs.argmax(1) == labels).sum().item()
        count     += labels.size(0)

        if ema is not None:
            ema.update()

        if pbar is not None:
            pbar.set_postfix(loss=f"{total_loss/count:.3f}",
                             acc=f"{correct/count:.3f}")

    return total_loss / count, correct / count


@torch.no_grad()
def validate(model, loader, criterion):
    """验证一个 epoch，返回 (avg_loss, accuracy)"""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct   += (outputs.argmax(1) == labels).sum().item()
        count     += labels.size(0)

    return total_loss / count, correct / count


def predict(img_path, model, class_names, use_tta=True):
    """预测单张图片 -> (类别名, 置信度, 各类概率)"""
    img = Image.open(img_path).convert('RGB')

    if use_tta:
        # 多次增强取平均
        probs_sum = None
        for tf in TTA_TRANSFORMS:
            t = tf(img).unsqueeze(0).to(DEVICE)
            p = torch.softmax(model(t), dim=1)
            probs_sum = p if probs_sum is None else probs_sum + p
        probs = probs_sum / len(TTA_TRANSFORMS)
    else:
        t      = val_transform(img).unsqueeze(0).to(DEVICE)
        probs  = torch.softmax(model(t), dim=1)

    idx  = probs.argmax(1).item()
    return class_names[idx], probs[0, idx].item(), probs[0].cpu().numpy()


# ╔══════════════════════════════════════════════════════════════╗
# ║              训 练 曲 线 图                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def draw_curves(all_history, save_path):
    """为每轮训练画 loss/acc 曲线"""
    if not HAS_PLOT:
        print("\n  [跳过] matplotlib 未安装，不生成曲线图")
        return

    n_loops = len(all_history)
    n_cols  = min(3, n_loops)
    n_rows  = math.ceil(n_loops / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 6, n_rows * 4.5))
    if n_rows == 1:
        axes = [axes]
    axes = np.atleast_1d(axes).flatten()

    for idx, hist in enumerate(all_history):
        ax  = axes[idx]
        ax2 = ax.twinx()
        ep  = range(1, len(hist['train_loss']) + 1)

        ax.plot(ep, hist['train_loss'], 'b-', alpha=0.5, lw=1.5, label='Train Loss')
        ax.plot(ep, hist['val_loss'],   'b-', alpha=1.0, lw=2.5, label='Val Loss')
        ax2.plot(ep, hist['train_acc'], 'r-', alpha=0.5, lw=1.5, label='Train Acc')
        ax2.plot(ep, hist['val_acc'],   'r-', alpha=1.0, lw=2.5, label='Val Acc')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color='b')
        ax2.set_ylabel('Accuracy', color='r')
        ax.set_title(f'Loop {hist["loop"]}')

        # 阶段分割线
        s1  = CONFIG['epochs_stage1']
        s2a = CONFIG['epochs_stage2a']
        for sep, lbl in [(s1, 'S1→S2a'), (s1 + s2a, 'S2a→S2b')]:
            if sep < len(ep):
                ax.axvline(sep + 0.5, color='gray', ls=':', alpha=0.5)
                y_top = ax.get_ylim()[1]
                ax.text(sep + 0.5, y_top * 0.95, lbl,
                        ha='center', fontsize=8, color='gray')

        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines],
                  loc='upper right', fontsize=7)

    # 隐藏多余子图
    for i in range(n_loops, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [曲线] {save_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║       解 冻 工 具                                             ║
# ╚══════════════════════════════════════════════════════════════╝

def unfreeze_layers(model, patterns):
    """按名称模式解冻参数"""
    for name, param in model.named_parameters():
        if any(p in name for p in patterns):
            param.requires_grad = True


# ╔══════════════════════════════════════════════════════════════╗
# ║                  主 程 序                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    start_time = time.time()

    # ── 1. 加载数据 ──
    print("\n" + "=" * 70)
    print("  RockClass V7 — ResNet50 渐进解冻三阶段训练")
    print("=" * 70)

    dataset = ImageFolder(str(DATA_DIR))
    class_names = dataset.classes
    num_classes = len(class_names)

    full_config['num_classes'] = num_classes
    full_config['class_names'] = class_names

    # 划分训练/验证集
    n_total    = len(dataset)
    n_val      = max(1, int(n_total * CONFIG['val_split']))
    n_train    = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    # 提取文件路径
    train_files = [(dataset.samples[i][0], dataset.samples[i][1])
                   for i in train_ds.indices]
    val_files   = [(dataset.samples[i][0], dataset.samples[i][1])
                   for i in val_ds.indices]

    # 轻量 Dataset 包装
    class FileDataset(Dataset):
        def __init__(self, files, transform):
            self.files     = files
            self.transform = transform
        def __len__(self): return len(self.files)
        def __getitem__(self, i):
            path, label = self.files[i]
            return self.transform(Image.open(path).convert('RGB')), label

    val_loader = DataLoader(
        FileDataset(val_files, val_transform),
        batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=True)

    print(f"  类别: {class_names}")
    print(f"  训练: {n_train}  验证: {n_val}")
    print(f"  策略: SGD+Nesterov | LabelSmoothing={CONFIG['label_smoothing']} | "
          f"渐进解冻(layer4→layer3)")

    # ── 2. 多轮训练 ──
    best_overall_acc = 0.0
    report_rows      = []
    all_history      = []

    for loop in range(1, CONFIG['total_loops'] + 1):
        print(f"\n{'─' * 70}")
        print(f"  {loop}/{CONFIG['total_loops']} 轮训练")
        print(f"{'─' * 70}")

        history = {'loop': loop,
                   'train_loss': [], 'train_acc': [],
                   'val_loss': [],   'val_acc': []}

        # 固定随机种子（可复现）
        seed = loop * 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        # DataLoader
        train_loader = DataLoader(
            FileDataset(train_files, train_transform),
            batch_size=CONFIG['batch_size'], shuffle=True,
            num_workers=CONFIG['num_workers'], pin_memory=True)

        # 构建模型
        print("  >>> 构建模型")
        model = build_model(num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
        scaler    = GradScaler() if AMP_ENABLED else None
        ema       = EMA(model, decay=CONFIG['ema_decay'])

        # ─── 阶段1: 训练分类头 ───
        print(f"\n  [阶段 1] 分类头训练 — {CONFIG['epochs_stage1']} epochs, lr={CONFIG['lr_stage1']}")
        optimizer = optim.SGD(model.parameters(),
                              lr=CONFIG['lr_stage1'],
                              momentum=CONFIG['momentum'],
                              weight_decay=CONFIG['weight_decay'],
                              nesterov=CONFIG['nesterov'])
        best_s1 = 0.0

        for ep in range(1, CONFIG['epochs_stage1'] + 1):
            pbar = None
            if HAS_TQDM:
                pbar = tqdm(train_loader, desc=f"  S1 E{ep:02d}",
                            leave=False, ncols=90)
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer,
                scaler=scaler, ema=ema, pbar=pbar)
            if pbar: pbar.close()

            val_loss, val_acc = validate(model, val_loader, criterion)
            best_s1 = max(best_s1, val_acc)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"    E{ep:02d}/{CONFIG['epochs_stage1']:02d}  |  "
                  f"t_loss: {train_loss:.4f}  t_acc: {train_acc:.4f}  |  "
                  f"v_loss: {val_loss:.4f}  v_acc: {val_acc:.4f}")

        # ─── 阶段2a: 解冻 layer4 ───
        print(f"\n  [阶段 2a] 解冻 layer4 — {CONFIG['epochs_stage2a']} epochs, lr={CONFIG['lr_stage2a']}")
        unfreeze_layers(model, ['layer4', 'fc'])
        ema       = EMA(model, decay=CONFIG['ema_decay'])
        optimizer = optim.SGD(model.parameters(),
                              lr=CONFIG['lr_stage2a'],
                              momentum=CONFIG['momentum'],
                              weight_decay=CONFIG['weight_decay'],
                              nesterov=CONFIG['nesterov'])
        best_s2a = 0.0

        for ep in range(1, CONFIG['epochs_stage2a'] + 1):
            pbar = None
            if HAS_TQDM:
                pbar = tqdm(train_loader, desc=f"  S2a E{ep:02d}",
                            leave=False, ncols=90)
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer,
                scaler=scaler, ema=ema, pbar=pbar)
            if pbar: pbar.close()

            val_loss, val_acc = validate(model, val_loader, criterion)
            best_s2a = max(best_s2a, val_acc)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"    E{ep:02d}/{CONFIG['epochs_stage2a']:02d}  |  "
                  f"t_loss: {train_loss:.4f}  t_acc: {train_acc:.4f}  |  "
                  f"v_loss: {val_loss:.4f}  v_acc: {val_acc:.4f}")

        # ─── 阶段2b: 解冻 layer3+4 ───
        print(f"\n  [阶段 2b] 解冻 layer3+4 — {CONFIG['epochs_stage2b']} epochs, lr={CONFIG['lr_stage2b']}")
        unfreeze_layers(model, ['layer3'])
        ema       = EMA(model, decay=CONFIG['ema_decay'])
        optimizer = optim.SGD(model.parameters(),
                              lr=CONFIG['lr_stage2b'],
                              momentum=CONFIG['momentum'],
                              weight_decay=CONFIG['weight_decay'],
                              nesterov=CONFIG['nesterov'])
        best_s2b = 0.0

        for ep in range(1, CONFIG['epochs_stage2b'] + 1):
            pbar = None
            if HAS_TQDM:
                pbar = tqdm(train_loader, desc=f"  S2b E{ep:02d}",
                            leave=False, ncols=90)
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer,
                scaler=scaler, ema=ema, pbar=pbar)
            if pbar: pbar.close()

            val_loss, val_acc = validate(model, val_loader, criterion)
            best_s2b = max(best_s2b, val_acc)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"    E{ep:02d}/{CONFIG['epochs_stage2b']:02d}  |  "
                  f"t_loss: {train_loss:.4f}  t_acc: {train_acc:.4f}  |  "
                  f"v_loss: {val_loss:.4f}  v_acc: {val_acc:.4f}")

        all_history.append(history)

        # 本轮最佳
        final_acc = max(best_s1, best_s2a, best_s2b, val_acc)
        report_rows.append({
            'loop':         loop,
            'best_s1':      f'{best_s1:.4f}',
            'best_s2a':     f'{best_s2a:.4f}',
            'best_s2b':     f'{best_s2b:.4f}',
            'final_acc':    f'{final_acc:.4f}',
        })

        # 保存最佳模型
        if final_acc > best_overall_acc:
            best_overall_acc = final_acc
            ema.apply()
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names':      class_names,
                'accuracy':         final_acc,
                'loop':             loop,
                'timestamp':        TIMESTAMP,
                'config':           full_config,
            }, MODEL_PATH)
            ema.restore()
            print(f"\n    ★ 新最佳！acc={final_acc:.4f}  →  {MODEL_PATH}")
        else:
            print(f"\n    acc={final_acc:.4f}  (最佳: {best_overall_acc:.4f})")

    # ── 3. 汇总报告 ──
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  训练完成！总耗时 {elapsed/60:.1f} 分钟")
    print(f"  最佳准确率: {best_overall_acc:.4f}")
    print(f"{'=' * 70}")

    # 打印表格
    hdr = f"  {'轮次':<6} {'S1最佳':<10} {'S2a最佳':<10} {'S2b最佳':<10} {'最终':<10}"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))
    for i, row in enumerate(report_rows):
        star = ' ★' if (i + 1) == max(
            range(len(report_rows)),
            key=lambda x: float(report_rows[x]['final_acc'])) + 1 else ''
        print(f"  {row['loop']:<6} {row['best_s1']:<10} "
              f"{row['best_s2a']:<10} {row['best_s2b']:<10} "
              f"{row['final_acc']:<10}{star}")

    # 保存 CSV
    import csv
    with open(REPORT_PATH, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['loop', 'best_s1', 'best_s2a',
                                          'best_s2b', 'final_acc'])
        w.writeheader()
        w.writerows(report_rows)

    # 保存配置
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(full_config, f, ensure_ascii=False, indent=2)

    # 画曲线
    draw_curves(all_history, CURVE_PATH)

    # 输出文件清单
    print(f"\n  输出文件:")
    print(f"    {MODEL_PATH}")
    print(f"    {REPORT_PATH}")
    print(f"    {CONFIG_PATH}")
    if HAS_PLOT:
        print(f"    {CURVE_PATH}")

    # ── 4. 交互预测 ──
    print(f"\n{'=' * 70}")
    print(f"  交互预测 — 请在弹出的窗口中选择图片（取消则退出）")
    print(f"{'=' * 70}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = build_model(num_classes, verbose=False).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    while True:
        path = filedialog.askopenfilename(
            title="选择岩石图片（取消退出）",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            print("  已退出。")
            break

        try:
            cls1, conf1, probs1 = predict(path, model, class_names, use_tta=False)
            cls2, conf2, probs2 = predict(path, model, class_names, use_tta=True)
        except Exception as e:
            print(f"  预测失败: {e}")
            continue

        print(f"\n  {'─' * 50}")
        print(f"  图片: {Path(path).name}")
        print(f"  普通: {cls1} ({conf1:.2%})")
        print(f"  TTA : {cls2} ({conf2:.2%})")
        for name, prob in zip(class_names, probs2):
            bar = '█' * int(prob * 20)
            print(f"    {name:<12} {prob:.2%}  {bar}")

    print(f"\n{'=' * 70}")
    print(f"  RockClass V7 全部完成")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
