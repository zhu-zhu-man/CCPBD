"""
RCF (Richer Convolutional Features) 模型训练与测试脚本
基于 VGG16 的多尺度边缘检测网络，在 CCPBD 数据集上进行边缘检测

使用方法:
    python train_rcf.py --data_root ./CCPBD --epochs 50 --lr 1e-4
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import get_dataloader
from models.edge_factory import get_edge_model
from utils.losses import WeightedBCELoss

# 模型配置
MODEL_NAME = 'rcf'
DISPLAY_NAME = 'RCF'


def calculate_ods_ois(all_preds, all_targets, thresholds=None, eps=1e-7):
    """计算 ODS 和 OIS 指标"""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    
    all_preds_flat = np.concatenate([p.flatten() for p in all_preds])
    all_targets_flat = np.concatenate([t.flatten() for t in all_targets])
    
    best_ods_f1 = 0
    ods_p, ods_r = 0, 0
    
    for t in thresholds:
        pred_binary = (all_preds_flat > t).astype(np.float32)
        tp = np.sum(pred_binary * all_targets_flat)
        fp = np.sum(pred_binary * (1 - all_targets_flat))
        fn = np.sum((1 - pred_binary) * all_targets_flat)
        
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        
        if f1 > best_ods_f1:
            best_ods_f1 = f1
            ods_p, ods_r = p, r
    
    best_f1_per_image = []
    for pred, target in zip(all_preds, all_targets):
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        best_f1 = 0
        
        for t in thresholds:
            pred_binary = (pred_flat > t).astype(np.float32)
            tp = np.sum(pred_binary * target_flat)
            fp = np.sum(pred_binary * (1 - target_flat))
            fn = np.sum((1 - pred_binary) * target_flat)
            
            p = tp / (tp + fp + eps)
            r = tp / (tp + fn + eps)
            f1 = 2 * p * r / (p + r + eps)
            
            if f1 > best_f1:
                best_f1 = f1
        
        best_f1_per_image.append(best_f1)
    
    ois_f1 = np.mean(best_f1_per_image)
    
    return {
        'ODS_F1': best_ods_f1 * 100,
        'ODS_Precision': ods_p * 100,
        'ODS_Recall': ods_r * 100,
        'OIS_F1': ois_f1 * 100
    }


def calculate_edge_loss(outputs, targets, criterion):
    """计算边缘检测损失"""
    if isinstance(outputs, dict):
        total_loss = 0
        count = 0
        for key, pred in outputs.items():
            if pred.shape[2:] != targets.shape[2:]:
                pred = F.interpolate(pred, size=targets.shape[2:], mode='bilinear', align_corners=False)
            
            if pred.max() <= 1 and pred.min() >= 0:
                loss = F.binary_cross_entropy(pred, targets)
            else:
                loss = criterion(pred, targets)
            total_loss += loss
            count += 1
        return total_loss / max(count, 1)
    else:
        if outputs.shape[2:] != targets.shape[2:]:
            outputs = F.interpolate(outputs, size=targets.shape[2:], mode='bilinear', align_corners=False)
        return criterion(outputs, targets)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
    
    for images, edges, _ in pbar:
        images = images.to(device)
        edges = edges.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = calculate_edge_loss(outputs, edges, criterion)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return {'loss': total_loss / num_batches}


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # 在GPU上向量化阈值计算，避免逐图逐阈值的Python循环
    thresholds = torch.linspace(0.05, 0.95, 19, device=device)
    eps = 1e-7
    tp_tot = torch.zeros_like(thresholds)
    fp_tot = torch.zeros_like(thresholds)
    fn_tot = torch.zeros_like(thresholds)
    best_f1_list: List[float] = []

    pbar = tqdm(dataloader, desc='Validating')

    for images, edges, _ in pbar:
        images = images.to(device)
        edges = edges.to(device)

        outputs = model(images)
        loss = calculate_edge_loss(outputs, edges, criterion)

        total_loss += loss.item()
        num_batches += 1

        # 选择融合的边缘预测
        if isinstance(outputs, dict):
            edge_pred = outputs.get('fuse', outputs.get('output', list(outputs.values())[-1]))
        else:
            edge_pred = outputs

        # 统一尺寸
        if edge_pred.shape[2:] != edges.shape[2:]:
            edge_pred = F.interpolate(edge_pred, size=edges.shape[2:], mode='bilinear', align_corners=False)

        # 统一通道维度为 [B,1,H,W]
        if edge_pred.dim() == 3:
            edge_pred = edge_pred.unsqueeze(1)

        # 确保为概率分布
        if (edge_pred.max() > 1) or (edge_pred.min() < 0):
            edge_pred = torch.sigmoid(edge_pred)

        targets_bin = (edges > 0.5).float()

        # [T,B,1,H,W] 的阈值向量化比较
        pred_bin = (edge_pred.unsqueeze(0) > thresholds[:, None, None, None, None]).float()

        # 批量统计每个阈值的 tp/fp/fn（先按图像求，再在B维聚合）
        tp_b = (pred_bin * targets_bin.unsqueeze(0)).sum(dim=(2, 3, 4))  # [T,B]
        fp_b = (pred_bin * (1 - targets_bin).unsqueeze(0)).sum(dim=(2, 3, 4))  # [T,B]
        fn_b = (((1 - pred_bin) * targets_bin.unsqueeze(0))).sum(dim=(2, 3, 4))  # [T,B]

        tp_tot += tp_b.sum(dim=1)
        fp_tot += fp_b.sum(dim=1)
        fn_tot += fn_b.sum(dim=1)

        # OIS：每张图在所有阈值上取最佳F1
        p_b = tp_b / (tp_b + fp_b + eps)
        r_b = tp_b / (tp_b + fn_b + eps)
        f1_b = 2 * p_b * r_b / (p_b + r_b + eps)  # [T,B]
        best_f1_batch = f1_b.max(dim=0).values  # [B]
        best_f1_list.extend(best_f1_batch.detach().cpu().tolist())

        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    # ODS：在所有阈值上取最佳F1（全数据）
    p = tp_tot / (tp_tot + fp_tot + eps)
    r = tp_tot / (tp_tot + fn_tot + eps)
    f1 = 2 * p * r / (p + r + eps)
    idx = torch.argmax(f1)

    metrics = {
        'ODS_F1': (f1[idx] * 100).item(),
        'ODS_Precision': (p[idx] * 100).item(),
        'ODS_Recall': (r[idx] * 100).item(),
        'OIS_F1': (np.mean(best_f1_list) * 100),
    }

    return {
        'loss': total_loss / max(num_batches, 1),
        'ods_f1': metrics['ODS_F1'] / 100,
        'ois_f1': metrics['OIS_F1'] / 100
    }


@torch.no_grad()
def test(model, dataloader, device):
    """测试模型"""
    model.eval()
    thresholds = torch.linspace(0.05, 0.95, 19, device=device)
    eps = 1e-7
    tp_tot = torch.zeros_like(thresholds)
    fp_tot = torch.zeros_like(thresholds)
    fn_tot = torch.zeros_like(thresholds)
    best_f1_list: List[float] = []

    pbar = tqdm(dataloader, desc='Testing')

    for images, edges, _ in pbar:
        images = images.to(device)
        edges = edges.to(device)

        outputs = model(images)

        if isinstance(outputs, dict):
            edge_pred = outputs.get('fuse', outputs.get('output', list(outputs.values())[-1]))
        else:
            edge_pred = outputs

        if edge_pred.shape[2:] != edges.shape[2:]:
            edge_pred = F.interpolate(edge_pred, size=edges.shape[2:], mode='bilinear', align_corners=False)

        if edge_pred.dim() == 3:
            edge_pred = edge_pred.unsqueeze(1)

        if (edge_pred.max() > 1) or (edge_pred.min() < 0):
            edge_pred = torch.sigmoid(edge_pred)

        targets_bin = (edges > 0.5).float()

        pred_bin = (edge_pred.unsqueeze(0) > thresholds[:, None, None, None, None]).float()

        tp_b = (pred_bin * targets_bin.unsqueeze(0)).sum(dim=(2, 3, 4))  # [T,B]
        fp_b = (pred_bin * (1 - targets_bin).unsqueeze(0)).sum(dim=(2, 3, 4))  # [T,B]
        fn_b = (((1 - pred_bin) * targets_bin.unsqueeze(0))).sum(dim=(2, 3, 4))  # [T,B]

        tp_tot += tp_b.sum(dim=1)
        fp_tot += fp_b.sum(dim=1)
        fn_tot += fn_b.sum(dim=1)

        p_b = tp_b / (tp_b + fp_b + eps)
        r_b = tp_b / (tp_b + fn_b + eps)
        f1_b = 2 * p_b * r_b / (p_b + r_b + eps)
        best_f1_batch = f1_b.max(dim=0).values
        best_f1_list.extend(best_f1_batch.detach().cpu().tolist())

    p = tp_tot / (tp_tot + fp_tot + eps)
    r = tp_tot / (tp_tot + fn_tot + eps)
    f1 = 2 * p * r / (p + r + eps)
    idx = torch.argmax(f1)

    results = {
        'ODS_F1': (f1[idx] * 100).item(),
        'ODS_Precision': (p[idx] * 100).item(),
        'ODS_Recall': (r[idx] * 100).item(),
        'OIS_F1': (np.mean(best_f1_list) * 100),
    }
    return results


def main(args):
    print("\n" + "=" * 60)
    print(f"{DISPLAY_NAME} Training & Evaluation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nLoading datasets from: {args.data_root}")
    
    train_loader = get_dataloader(
        root_dir=args.data_root, split='train', task='edge',
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    
    val_loader = get_dataloader(
        root_dir=args.data_root, split='val', task='edge',
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"Val samples: {len(val_loader.dataset)}")
    
    test_loader = get_dataloader(
        root_dir=args.data_root, split='test', task='edge',
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    print(f"\nCreating {DISPLAY_NAME} model with VGG16 pretrained weights...")
    model = get_edge_model(model_name=MODEL_NAME, pretrained=True)
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    criterion = WeightedBCELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    best_model_path = os.path.join(args.output_dir, f'{MODEL_NAME}_best.pth')
    best_val_f1 = 0.0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, ODS F1: {val_metrics['ods_f1']:.4f}, OIS F1: {val_metrics['ois_f1']:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if val_metrics['ods_f1'] > best_val_f1:
            best_val_f1 = val_metrics['ods_f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ods_f1': best_val_f1,
            }, best_model_path)
            print(f"  ✓ New best model saved! Val ODS F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\n{'='*60}")
    print("Loading best model for testing...")
    print(f"{'='*60}")
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with Val ODS F1: {checkpoint['val_ods_f1']:.4f}")
    
    test_results = test(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"{DISPLAY_NAME} Test Results:")
    print(f"{'='*60}")
    print(f"  ODS F1:        {test_results['ODS_F1']:.2f}%")
    print(f"  ODS Precision: {test_results['ODS_Precision']:.2f}%")
    print(f"  ODS Recall:    {test_results['ODS_Recall']:.2f}%")
    print(f"  OIS F1:        {test_results['OIS_F1']:.2f}%")
    print(f"{'='*60}")
    
    results_path = os.path.join(args.output_dir, f'{MODEL_NAME}_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"{DISPLAY_NAME} Test Results\n")
        f.write(f"{'='*40}\n")
        for k, v in test_results.items():
            f.write(f"{k}: {v:.4f}%\n")
    print(f"\nResults saved to: {results_path}")
    
    return test_results


def parse_args():
    parser = argparse.ArgumentParser(description=f'{DISPLAY_NAME} Training')
    parser.add_argument('--data_root', type=str, default='./CCPBD')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
