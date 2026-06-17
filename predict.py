# -*- coding: utf-8 -*-
"""
RockClass Predict — 岩石薄片图像分类预测工具
=============================================
自动查找 runs/ 下最新训练的模型，提供交互式预测。

使用:
    python predict.py              → 交互模式（弹窗选图）
    python predict.py image.jpg    → 命令行预测单张图片
    python predict.py folder/      → 批量预测文件夹内所有图片
"""

import os
import sys
from pathlib import Path
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE  = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SCRIPT_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────

def find_model():
    """查找最新训练的模型文件"""
    candidates = []

    # 1. runs/<latest>/best_model.pt
    runs_dir = SCRIPT_DIR / 'runs'
    if runs_dir.is_dir():
        for d in sorted(runs_dir.iterdir(), reverse=True):
            p = d / 'best_model.pt'
            if p.exists():
                candidates.append((p.stat().st_mtime, p))

    # 2. 同目录下的 rock_best_model_v5.pt（兼容旧版）
    legacy = SCRIPT_DIR / 'rock_best_model_v5.pt'
    if legacy.exists():
        candidates.append((legacy.stat().st_mtime, legacy))

    # 3. 命令行指定的路径
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.suffix in ('.pt', '.pth') and p.exists():
            candidates.append((p.stat().st_mtime, p))

    if not candidates:
        print("\n  [错误] 未找到模型文件！")
        print(f"  请先运行 train.py 进行训练。")
        print(f"  预期位置: {runs_dir}/YYYYmmdd_HHMMSS/best_model.pt")
        sys.exit(1)

    # 返回最新的
    candidates.sort(reverse=True)
    return candidates[0][1]


def build_model(num_classes):
    """构建与训练时一致的模型结构"""
    model = models.resnet50(weights=None)
    for p in model.parameters():
        p.requires_grad = False
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


def load_model(model_path):
    """加载模型 -> (model, class_names, accuracy)"""
    print(f"  [加载] {model_path}")
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    class_names = ckpt['class_names']
    acc         = ckpt.get('accuracy', ckpt.get('val_acc', '?'))

    model = build_model(len(class_names)).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, class_names, acc


# ─────────────────────────────────────────────
# 预处理 & 预测
# ─────────────────────────────────────────────

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# TTA 变换
TTA_TF = [
    val_tf,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
]


@torch.no_grad()
def predict(img_path, model, class_names, use_tta=True):
    """预测图片，返回 (类别, 置信度, 概率数组)"""
    img = Image.open(img_path).convert('RGB')

    if use_tta:
        probs_sum = None
        for tf in TTA_TF:
            t = tf(img).unsqueeze(0).to(DEVICE)
            p = torch.softmax(model(t), dim=1)
            probs_sum = p if probs_sum is None else probs_sum + p
        probs = probs_sum / len(TTA_TF)
    else:
        t = val_tf(img).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(model(t), dim=1)

    idx = probs.argmax(1).item()
    return class_names[idx], probs[0, idx].item(), probs[0].cpu().numpy()


def print_result(path, cls_name, conf, probs, class_names):
    """格式化打印预测结果"""
    print(f"\n  {'─' * 50}")
    print(f"  图片: {Path(path).name}")
    print(f"  预测: {cls_name}  (置信度 {conf:.2%})")
    print(f"  {'─' * 50}")
    for name, p in zip(class_names, probs):
        bar = '█' * int(p * 20)
        print(f"    {name:<12} {p:.2%}  {bar}")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def main():
    print(f"\n  RockClass Predict")
    print(f"  设备: {DEVICE}")

    model_path = find_model()
    model, class_names, acc = load_model(model_path)
    print(f"  类别: {class_names}")
    print(f"  准确率: {acc}")

    # ── 收集要预测的图片 ──
    targets = []

    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.suffix in ('.pt', '.pth'):
            continue  # 模型文件已处理
        if p.is_dir():
            targets.extend(sorted(p.glob('*')))
        elif p.is_file():
            targets.append(p)

    if targets:
        # 命令行模式：批量预测
        print(f"\n  共 {len(targets)} 张图片\n")
        for p in targets:
            if p.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                continue
            try:
                cls_name, conf, probs = predict(p, model, class_names, use_tta=True)
                print_result(p, cls_name, conf, probs, class_names)
            except Exception as e:
                print(f"  [跳过] {p.name}: {e}")
        return

    # ── 交互模式：弹窗选图 ──
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    print("\n  请在弹出窗口中选择图片（取消退出）...")

    while True:
        path = filedialog.askopenfilename(
            title="选择岩石图片（取消退出）",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            print("  已退出。")
            break
        try:
            _, _, probs_tt = predict(path, model, class_names, use_tta=False)
            cls_name, conf, probs_t = predict(path, model, class_names, use_tta=True)
            print_result(path, cls_name, conf, probs_t, class_names)
        except Exception as e:
            print(f"  [错误] {e}")


if __name__ == '__main__':
    main()
