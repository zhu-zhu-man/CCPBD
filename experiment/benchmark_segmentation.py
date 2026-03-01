"""
CCPBD Segmentation Benchmark
一键运行所有分割模型的训练与测试脚本

使用方法:
    # 训练并测试所有模型
    python benchmark_segmentation.py --data_root ./CCPBD --output_dir ./results
    
    # 仅测试（跳过训练）
    python benchmark_segmentation.py --data_root ./CCPBD --skip_train
    
    # 训练单个模型
    python benchmark_segmentation.py --model unet --data_root ./CCPBD

指标:
    - Pixel Accuracy (PA)
    - Precision
    - Recall  
    - F1-Score
    - IoU (Intersection over Union)
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

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
from models.segmentation_factory import get_segmentation_model
from utils.losses import BCEDiceLoss

# 尝试导入表格显示库
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ============================================================================
# 模型配置
# ============================================================================
MODELS = {
    'UNet': {
        'name': 'unet',
        'encoder': 'resnet34',
        'display_name': 'UNet'
    },
    'DeepLabV3+': {
        'name': 'deeplabv3plus',
        'encoder': 'resnet50',
        'display_name': 'DeepLabV3+'
    },
    'SegFormer': {
        'name': 'segformer',
        'encoder': 'mit_b2',
        'display_name': 'SegFormer'
    },
    'VM-UNet': {
        'name': 'vmunet',
        'encoder': None,
        'display_name': 'VM-UNet'
    }
}


# ============================================================================
# 评价指标计算
# ============================================================================
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


# ============================================================================
# 训练函数
# ============================================================================
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
    
    pbar = tqdm(dataloader, desc='Validating', leave=False)
    
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
    
    pbar = tqdm(dataloader, desc='Testing', leave=False)
    
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
    
    # 计算平均指标 (百分比)
    results = {k: np.mean(v) * 100 for k, v in all_metrics.items()}
    return results


# ============================================================================
# 单模型训练流程
# ============================================================================
def train_model(
    model_key: str,
    model_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args,
    device: torch.device
) -> Dict[str, float]:
    """训练并测试单个模型"""
    
    display_name = model_config['display_name']
    model_name = model_config['name']
    encoder = model_config['encoder']
    
    print(f"\n{'='*60}")
    print(f"Training {display_name}")
    print(f"{'='*60}")
    
    # 创建模型
    model = get_segmentation_model(
        model_name=model_name,
        num_classes=1,
        weights='imagenet',
        encoder_name=encoder
    )
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    # 损失函数和优化器
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # 训练
    best_model_path = os.path.join(args.output_dir, f'{model_name}_best.pth')
    best_val_iou = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_metrics['loss']:.4f}, Train IoU: {train_metrics['iou']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Val IoU: {val_metrics['iou']:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
            }, best_model_path)
            print(f"  ★ New best model saved! Val IoU: {best_val_iou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping triggered after {epoch+1} epochs")
                break
    
    # 加载最佳模型进行测试
    print(f"\nLoading best model for testing...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with Val IoU: {checkpoint['val_iou']:.4f}")
    
    # 测试
    test_results = test(model, test_loader, device, threshold=args.threshold)
    test_results['Params(M)'] = params
    
    print(f"\n{display_name} Test Results:")
    print(f"  Pixel Accuracy: {test_results['pixel_accuracy']:.2f}%")
    print(f"  Precision:      {test_results['precision']:.2f}%")
    print(f"  Recall:         {test_results['recall']:.2f}%")
    print(f"  F1-Score:       {test_results['f1']:.2f}%")
    print(f"  IoU:            {test_results['iou']:.2f}%")
    
    return test_results


def test_model(
    model_key: str,
    model_config: dict,
    test_loader: DataLoader,
    args,
    device: torch.device
) -> Dict[str, float]:
    """仅测试单个模型（加载预训练权重）"""
    
    display_name = model_config['display_name']
    model_name = model_config['name']
    encoder = model_config['encoder']
    
    print(f"\n[{display_name}]")
    
    # 创建模型
    model = get_segmentation_model(
        model_name=model_name,
        num_classes=1,
        weights='imagenet',
        encoder_name=encoder
    )
    
    # 加载权重
    weights_path = os.path.join(args.output_dir, f'{model_name}_best.pth')
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded weights from: {weights_path}")
    else:
        print(f"  Warning: No weights found at {weights_path}, using ImageNet pretrained")
    
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.2f}M")
    
    # 测试
    test_results = test(model, test_loader, device, threshold=args.threshold)
    test_results['Params(M)'] = params
    
    print(f"  ✓ IoU: {test_results['iou']:.2f}%  |  F1: {test_results['f1']:.2f}%")
    
    return test_results


# ============================================================================
# 结果报告
# ============================================================================
def print_results_table(results: Dict[str, Dict[str, float]]):
    """打印格式化的结果表格"""
    
    print("\n" + "=" * 80)
    print("SEGMENTATION BENCHMARK RESULTS")
    print("=" * 80)
    
    if HAS_PANDAS:
        df = pd.DataFrame(results).T
        df = df.round(2)
        
        # 重新排列列顺序
        cols = ['pixel_accuracy', 'precision', 'recall', 'f1', 'iou', 'Params(M)']
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        # 重命名列
        df.columns = ['PA(%)', 'Prec(%)', 'Recall(%)', 'F1(%)', 'IoU(%)', 'Params(M)']
        
        print(df.to_string())
        print("=" * 80)
        return df
    
    elif HAS_TABULATE:
        headers = ['Model', 'PA(%)', 'Prec(%)', 'Recall(%)', 'F1(%)', 'IoU(%)', 'Params(M)']
        rows = []
        
        for model_name, metrics in results.items():
            row = [
                model_name,
                f"{metrics.get('pixel_accuracy', 0):.2f}",
                f"{metrics.get('precision', 0):.2f}",
                f"{metrics.get('recall', 0):.2f}",
                f"{metrics.get('f1', 0):.2f}",
                f"{metrics.get('iou', 0):.2f}",
                f"{metrics.get('Params(M)', 0):.2f}"
            ]
            rows.append(row)
        
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print("=" * 80)
    
    else:
        print(f"{'Model':<15} {'PA(%)':<10} {'Prec(%)':<10} {'Recall(%)':<10} "
              f"{'F1(%)':<10} {'IoU(%)':<10} {'Params(M)':<10}")
        print("-" * 80)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<15} "
                  f"{metrics.get('pixel_accuracy', 0):<10.2f} "
                  f"{metrics.get('precision', 0):<10.2f} "
                  f"{metrics.get('recall', 0):<10.2f} "
                  f"{metrics.get('f1', 0):<10.2f} "
                  f"{metrics.get('iou', 0):<10.2f} "
                  f"{metrics.get('Params(M)', 0):<10.2f}")
        
        print("=" * 80)


def save_results_csv(results: Dict[str, Dict[str, float]], output_path: str):
    """保存结果到 CSV 文件"""
    
    if HAS_PANDAS:
        df = pd.DataFrame(results).T
        df.index.name = 'Model'
        df = df.round(4)
        df.to_csv(output_path)
        print(f"\n✓ Results saved to: {output_path}")
    else:
        with open(output_path, 'w') as f:
            metrics_keys = list(list(results.values())[0].keys())
            f.write('Model,' + ','.join(metrics_keys) + '\n')
            
            for model_name, metrics in results.items():
                values = [f"{metrics[k]:.4f}" for k in metrics_keys]
                f.write(f"{model_name}," + ','.join(values) + '\n')
        
        print(f"\n✓ Results saved to: {output_path}")


# ============================================================================
# 主函数
# ============================================================================
def main(args):
    """主函数"""
    
    print("\n" + "=" * 80)
    print("CCPBD SEGMENTATION BENCHMARK")
    print("Chinese Cropland Parcel and Boundary Dataset - Technical Validation")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定要运行的模型
    if args.model:
        # 单个模型
        model_keys = [k for k in MODELS.keys() if MODELS[k]['name'] == args.model.lower()]
        if not model_keys:
            print(f"Error: Unknown model '{args.model}'")
            print(f"Available models: {[MODELS[k]['name'] for k in MODELS.keys()]}")
            return
    else:
        # 所有模型
        model_keys = list(MODELS.keys())
    
    print(f"Models to benchmark: {model_keys}")
    
    # 加载数据集
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
    
    # 存储所有结果
    all_results = {}
    
    # 遍历所有模型
    for model_key in model_keys:
        model_config = MODELS[model_key]
        
        try:
            if args.skip_train:
                # 仅测试
                results = test_model(
                    model_key, model_config, test_loader, args, device
                )
            else:
                # 训练并测试
                results = train_model(
                    model_key, model_config, 
                    train_loader, val_loader, test_loader, 
                    args, device
                )
            
            all_results[model_key] = results
            
        except Exception as e:
            print(f"  ✗ Error with {model_key}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_key] = {
                'pixel_accuracy': 0, 'precision': 0, 'recall': 0,
                'f1': 0, 'iou': 0, 'Error': str(e)
            }
    
    # 打印结果表格
    print_results_table(all_results)
    
    # 保存结果
    csv_path = os.path.join(args.output_dir, 'segmentation_benchmark.csv')
    save_results_csv(all_results, csv_path)
    
    print("\n✓ Benchmark completed!")
    
    return all_results


# ============================================================================
# 命令行参数
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='CCPBD Segmentation Benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='./CCPBD',
                        help='Dataset root directory')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    
    # 模型选择
    parser.add_argument('--model', type=str, default=None,
                        help='Train specific model (unet, deeplabv3plus, segformer, vmunet)')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training and only run testing')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
