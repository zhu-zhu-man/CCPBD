"""
VM-UNet 模型训练与测试脚本
基于 Mamba State Space Model 的分割网络，在 CCPBD 数据集上进行语义分割

使用方法:
    python train_vmunet.py --data_root ./CCPBD --epochs 50 --lr 1e-4
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import get_dataloader
from models.segmentation_factory import get_segmentation_model
from utils.losses import BCEDiceLoss

# 模型配置
MODEL_NAME = 'vmunet'
ENCODER_NAME = None
DISPLAY_NAME = 'VM-UNet'


def calculate_metrics(pred, target, threshold=0.5, eps=1e-7):
    """计算分割指标"""
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    tp = ((pred_binary == 1) & (target_binary == 1)).sum().item()
    fp = ((pred_binary == 1) & (target_binary == 0)).sum().item()
    fn = ((pred_binary == 0) & (target_binary == 1)).sum().item()
    tn = ((pred_binary == 0) & (target_binary == 0)).sum().item()
    
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    pa = (tp + tn) / (tp + fp + fn + tn + eps)
    
    return {
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'pixel_accuracy': pa
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
    
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        with torch.no_grad():
            pred_prob = torch.sigmoid(outputs)
            metrics = calculate_metrics(pred_prob, masks)
            total_iou += metrics['iou']
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{metrics["iou"]:.4f}'
        })
    
    return {
        'loss': total_loss / num_batches,
        'iou': total_iou / num_batches
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Validating')
    
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        num_batches += 1
        
        pred_prob = torch.sigmoid(outputs)
        metrics = calculate_metrics(pred_prob, masks)
        total_iou += metrics['iou']
        total_f1 += metrics['f1']
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{metrics["iou"]:.4f}'
        })
    
    return {
        'loss': total_loss / num_batches,
        'iou': total_iou / num_batches,
        'f1': total_f1 / num_batches
    }


@torch.no_grad()
def test(model, dataloader, device, threshold=0.5):
    """测试模型"""
    model.eval()
    
    all_metrics = {
        'iou': [], 'f1': [], 'precision': [], 'recall': [], 'pixel_accuracy': []
    }
    
    pbar = tqdm(dataloader, desc='Testing')
    
    for images, masks, filenames in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        
        if outputs.max() > 1 or outputs.min() < 0:
            outputs = torch.sigmoid(outputs)
        
        for i in range(images.size(0)):
            metrics = calculate_metrics(outputs[i:i+1], masks[i:i+1], threshold)
            for k, v in metrics.items():
                all_metrics[k].append(v)
        
        pbar.set_postfix({'IoU': f'{np.mean(all_metrics["iou"]):.4f}'})
    
    results = {k: np.mean(v) * 100 for k, v in all_metrics.items()}
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
        root_dir=args.data_root, split='train', task='segmentation',
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    
    val_loader = get_dataloader(
        root_dir=args.data_root, split='val', task='segmentation',
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"Val samples: {len(val_loader.dataset)}")
    
    test_loader = get_dataloader(
        root_dir=args.data_root, split='test', task='segmentation',
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    print(f"\nCreating {DISPLAY_NAME} model...")
    model = get_segmentation_model(
        model_name=MODEL_NAME,
        num_classes=1,
        weights='imagenet',
        encoder_name=ENCODER_NAME
    )
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    best_model_path = os.path.join(args.output_dir, f'{MODEL_NAME}_best.pth')
    best_val_iou = 0.0
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
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
                'val_f1': val_metrics['f1'],
            }, best_model_path)
            print(f"  ✓ New best model saved! Val IoU: {best_val_iou:.4f}")
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
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with Val IoU: {checkpoint['val_iou']:.4f}")
    
    test_results = test(model, test_loader, device, threshold=args.threshold)
    
    print(f"\n{'='*60}")
    print(f"{DISPLAY_NAME} Test Results:")
    print(f"{'='*60}")
    print(f"  Pixel Accuracy: {test_results['pixel_accuracy']:.2f}%")
    print(f"  Precision:      {test_results['precision']:.2f}%")
    print(f"  Recall:         {test_results['recall']:.2f}%")
    print(f"  F1-Score:       {test_results['f1']:.2f}%")
    print(f"  IoU:            {test_results['iou']:.2f}%")
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
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
