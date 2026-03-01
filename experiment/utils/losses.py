"""
CCPBD Benchmark Utilities
通用的损失函数和训练工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ============================================================================
# 分割任务损失函数
# ============================================================================
class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        loss = focal_weight * bce
        
        return loss.mean()


# ============================================================================
# 边缘检测损失函数
# ============================================================================
class WeightedBCELoss(nn.Module):
    """
    Weighted BCE Loss for edge detection
    给边缘像素更高的权重，处理类别不平衡
    """
    
    def __init__(self, pos_weight: float = None):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 自动计算正样本权重
        if self.pos_weight is None:
            num_pos = target.sum()
            num_neg = target.numel() - num_pos
            pos_weight = num_neg / (num_pos + 1e-7)
        else:
            pos_weight = self.pos_weight
        
        # 计算加权 BCE
        loss = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=torch.tensor([pos_weight], device=pred.device)
        )
        
        return loss


class MultiScaleEdgeLoss(nn.Module):
    """
    Multi-scale edge detection loss
    用于 HED, RCF 等多尺度输出的模型
    """
    
    def __init__(self, num_scales: int = 5):
        super().__init__()
        self.num_scales = num_scales
        self.bce = WeightedBCELoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        total_loss = 0
        count = 0
        
        for key, pred in outputs.items():
            if key.startswith('side') or key == 'fuse':
                # 调整预测尺寸匹配目标
                if pred.shape[2:] != target.shape[2:]:
                    pred = F.interpolate(
                        pred, size=target.shape[2:],
                        mode='bilinear', align_corners=False
                    )
                
                # pred 已经是 sigmoid 后的值
                if pred.max() <= 1 and pred.min() >= 0:
                    loss = F.binary_cross_entropy(pred, target)
                else:
                    loss = self.bce(pred, target)
                
                total_loss += loss
                count += 1
        
        return total_loss / max(count, 1)


# ============================================================================
# 学习率调度器
# ============================================================================
def get_scheduler(optimizer, scheduler_type: str, **kwargs):
    """获取学习率调度器"""
    
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'warmup_cosine':
        from torch.optim.lr_scheduler import LambdaLR
        
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        total_epochs = kwargs.get('total_epochs', 100)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ============================================================================
# 其他工具
# ============================================================================
class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class AverageMeter:
    """计算和存储平均值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
