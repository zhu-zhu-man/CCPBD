"""
CCPBD Edge Detection Benchmark
一键运行所有边缘检测模型的训练与测试脚本

使用方法:
    # 训练并测试所有模型
    python benchmark_edge.py --data_root ./CCPBD --output_dir ./results
    
    # 仅测试（跳过训练）
    python benchmark_edge.py --data_root ./CCPBD --skip_train
    
    # 训练单个模型
    python benchmark_edge.py --model hed --data_root ./CCPBD

指标:
    - ODS (Optimal Dataset Scale): 在整个数据集上找最优阈值
    - OIS (Optimal Image Scale): 每张图像单独找最优阈值
    - Precision / Recall
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
from models.edge_factory import get_edge_model
from utils.losses import WeightedBCELoss

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
    'HED': {
        'name': 'hed',
        'display_name': 'HED'
    },
    'RCF': {
        'name': 'rcf',
        'display_name': 'RCF'
    },
    'BDCN': {
        'name': 'bdcn',
        'display_name': 'BDCN'
    },
    'DexiNed': {
        'name': 'dexined',
        'display_name': 'DexiNed'
    }
}


# ============================================================================
# 评价指标计算
# ============================================================================
def calculate_ods_ois(all_preds, all_targets, thresholds=None, eps=1e-7):
    """计算 ODS 和 OIS 指标"""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    
    all_preds_flat = np.concatenate([p.flatten() for p in all_preds])
    all_targets_flat = np.concatenate([t.flatten() for t in all_targets])
    
    # ODS: 全局最优阈值
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
            ods_p = p
            ods_r = r
    
    # OIS: 每张图像最优
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
    """计算边缘检测损失（支持多尺度输出）"""
    if isinstance(outputs, dict):
        total_loss = 0
        count = 0
        for key, pred in outputs.items():
            if pred.shape[2:] != targets.shape[2:]:
                pred = F.interpolate(pred, size=targets.shape[2:], mode='bilinear', align_corners=False)
            
            # 判断是否已经是 sigmoid 后的值
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


# ============================================================================
# 训练函数
# ============================================================================
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
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc='Validating', leave=False)
    
    for images, edges, _ in pbar:
        images = images.to(device)
        edges = edges.to(device)
        
        outputs = model(images)
        loss = calculate_edge_loss(outputs, edges, criterion)
        
        total_loss += loss.item()
        num_batches += 1
        
        # 获取融合输出
        if isinstance(outputs, dict):
            edge_pred = outputs.get('fuse', outputs.get('output', list(outputs.values())[-1]))
        else:
            edge_pred = outputs
        
        all_preds.append(edge_pred.cpu().numpy())
        all_targets.append((edges > 0.5).float().cpu().numpy())
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # 计算 ODS F1
    all_preds_list = [p.squeeze() for batch in all_preds for p in batch]
    all_targets_list = [t.squeeze() for batch in all_targets for t in batch]
    
    metrics = calculate_ods_ois(all_preds_list, all_targets_list)
    
    return {
        'loss': total_loss / num_batches,
        'ods_f1': metrics['ODS_F1'] / 100,
        'ois_f1': metrics['OIS_F1'] / 100
    }


@torch.no_grad()
def test(model, dataloader, device):
    """测试模型"""
    model.eval()
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc='Testing', leave=False)
    
    for images, edges, _ in pbar:
        images = images.to(device)
        edges = edges.to(device)
        
        outputs = model(images)
        
        # 获取融合输出
        if isinstance(outputs, dict):
            edge_pred = outputs.get('fuse', outputs.get('output', list(outputs.values())[-1]))
        else:
            edge_pred = outputs
        
        all_preds.append(edge_pred.cpu().numpy())
        all_targets.append((edges > 0.5).float().cpu().numpy())
    
    # 展开为单张图像列表
    all_preds_list = [p.squeeze() for batch in all_preds for p in batch]
    all_targets_list = [t.squeeze() for batch in all_targets for t in batch]
    
    # 计算指标
    results = calculate_ods_ois(all_preds_list, all_targets_list)
    
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
    
    print(f"\n{'='*60}")
    print(f"Training {display_name}")
    print(f"{'='*60}")
    
    # 创建模型
    model = get_edge_model(model_name=model_name, pretrained=True)
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    # 损失函数和优化器
    criterion = WeightedBCELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # 训练
    best_model_path = os.path.join(args.output_dir, f'{model_name}_best.pth')
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Val ODS F1: {val_metrics['ods_f1']:.4f} | "
              f"LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_metrics['ods_f1'] > best_val_f1:
            best_val_f1 = val_metrics['ods_f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ods_f1': best_val_f1,
            }, best_model_path)
            print(f"  ★ New best model saved! Val ODS F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping triggered after {epoch+1} epochs")
                break
    
    # 加载最佳模型进行测试
    print(f"\nLoading best model for testing...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with Val ODS F1: {checkpoint['val_ods_f1']:.4f}")
    
    # 测试
    test_results = test(model, test_loader, device)
    test_results['Params(M)'] = params
    
    print(f"\n{display_name} Test Results:")
    print(f"  ODS F1:        {test_results['ODS_F1']:.2f}%")
    print(f"  ODS Precision: {test_results['ODS_Precision']:.2f}%")
    print(f"  ODS Recall:    {test_results['ODS_Recall']:.2f}%")
    print(f"  OIS F1:        {test_results['OIS_F1']:.2f}%")
    
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
    
    print(f"\n[{display_name}]")
    
    # 创建模型
    model = get_edge_model(model_name=model_name, pretrained=True)
    
    # 加载权重
    weights_path = os.path.join(args.output_dir, f'{model_name}_best.pth')
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded weights from: {weights_path}")
    else:
        print(f"  Warning: No weights found at {weights_path}, using VGG16 pretrained")
    
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.2f}M")
    
    # 测试
    test_results = test(model, test_loader, device)
    test_results['Params(M)'] = params
    
    print(f"  ✓ ODS F1: {test_results['ODS_F1']:.2f}%  |  OIS F1: {test_results['OIS_F1']:.2f}%")
    
    return test_results


# ============================================================================
# 结果报告
# ============================================================================
def print_results_table(results: Dict[str, Dict[str, float]]):
    """打印格式化的结果表格"""
    
    print("\n" + "=" * 80)
    print("EDGE DETECTION BENCHMARK RESULTS")
    print("=" * 80)
    
    if HAS_PANDAS:
        df = pd.DataFrame(results).T
        df = df.round(2)
        
        # 重新排列列顺序
        cols = ['ODS_F1', 'ODS_Precision', 'ODS_Recall', 'OIS_F1', 'Params(M)']
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        print(df.to_string())
        print("=" * 80)
        return df
    
    elif HAS_TABULATE:
        headers = ['Model', 'ODS F1(%)', 'ODS Prec(%)', 'ODS Rec(%)', 'OIS F1(%)', 'Params(M)']
        rows = []
        
        for model_name, metrics in results.items():
            row = [
                model_name,
                f"{metrics.get('ODS_F1', 0):.2f}",
                f"{metrics.get('ODS_Precision', 0):.2f}",
                f"{metrics.get('ODS_Recall', 0):.2f}",
                f"{metrics.get('OIS_F1', 0):.2f}",
                f"{metrics.get('Params(M)', 0):.2f}"
            ]
            rows.append(row)
        
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print("=" * 80)
    
    else:
        print(f"{'Model':<15} {'ODS F1(%)':<12} {'ODS Prec(%)':<12} "
              f"{'ODS Rec(%)':<12} {'OIS F1(%)':<12} {'Params(M)':<10}")
        print("-" * 80)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<15} "
                  f"{metrics.get('ODS_F1', 0):<12.2f} "
                  f"{metrics.get('ODS_Precision', 0):<12.2f} "
                  f"{metrics.get('ODS_Recall', 0):<12.2f} "
                  f"{metrics.get('OIS_F1', 0):<12.2f} "
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
    print("CCPBD EDGE DETECTION BENCHMARK")
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
                'ODS_F1': 0, 'ODS_Precision': 0, 'ODS_Recall': 0,
                'OIS_F1': 0, 'Error': str(e)
            }
    
    # 打印结果表格
    print_results_table(all_results)
    
    # 保存结果
    csv_path = os.path.join(args.output_dir, 'edge_benchmark.csv')
    save_results_csv(all_results, csv_path)
    
    print("\n✓ Benchmark completed!")
    
    return all_results


# ============================================================================
# 命令行参数
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='CCPBD Edge Detection Benchmark',
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
    
    # 模型选择
    parser.add_argument('--model', type=str, default=None,
                        help='Train specific model (hed, rcf, bdcn, dexined)')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training and only run testing')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
