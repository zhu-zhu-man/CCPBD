"""
Edge Detection Model Factory
提供统一接口创建边缘检测模型：HED, RCF, BDCN, DexiNed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

# 尝试导入预训练 VGG
try:
    from torchvision.models import vgg16, VGG16_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# ============================================================================
# 支持的模型列表
# ============================================================================
SUPPORTED_MODELS = ['hed', 'rcf', 'bdcn', 'dexined']


# ============================================================================
# 工具模块
# ============================================================================
class ConvBnRelu(nn.Module):
    """Conv-BN-ReLU 基础模块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        use_bn: bool = True,
        use_relu: bool = True
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=not use_bn
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=False) if use_relu else nn.Identity()  # 禁用 inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class SideOutput(nn.Module):
    """边缘检测的侧输出模块"""
    
    def __init__(self, in_channels: int, scale_factor: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.scale_factor = scale_factor
    
    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        x = self.conv(x)
        
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        elif self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        return x


# ============================================================================
# HED: Holistically-nested Edge Detection
# ============================================================================
class HED(nn.Module):
    """
    HED: Holistically-nested Edge Detection
    
    论文: Holistically-Nested Edge Detection (ICCV 2015)
    基于 VGG16 的多尺度边缘检测网络
    
    Args:
        pretrained: 是否使用预训练的 VGG16 权重
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # VGG16 特征提取器
        if HAS_TORCHVISION and pretrained:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            features = list(vgg.features.children())
            # 将所有 ReLU 的 inplace 设置为 False，避免梯度计算冲突
            features = self._disable_inplace_relu(features)
        else:
            features = self._make_vgg_features()
        
        # Stage 1: conv1_1, conv1_2, pool1 (64 channels)
        self.stage1 = nn.Sequential(*features[:4])
        
        # Stage 2: conv2_1, conv2_2, pool2 (128 channels)
        self.stage2 = nn.Sequential(*features[4:9])
        
        # Stage 3: conv3_1, conv3_2, conv3_3, pool3 (256 channels)
        self.stage3 = nn.Sequential(*features[9:16])
        
        # Stage 4: conv4_1, conv4_2, conv4_3, pool4 (512 channels)
        self.stage4 = nn.Sequential(*features[16:23])
        
        # Stage 5: conv5_1, conv5_2, conv5_3 (512 channels, no pool)
        self.stage5 = nn.Sequential(*features[23:30])
        
        # Side outputs
        self.side1 = SideOutput(64, scale_factor=1)
        self.side2 = SideOutput(128, scale_factor=2)
        self.side3 = SideOutput(256, scale_factor=4)
        self.side4 = SideOutput(512, scale_factor=8)
        self.side5 = SideOutput(512, scale_factor=16)
        
        # Fusion layer
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)
        
        # 初始化融合层
        nn.init.constant_(self.fuse.weight, 0.2)
        nn.init.constant_(self.fuse.bias, 0)
    
    def _make_vgg_features(self) -> list:
        """创建 VGG 特征提取层 (无预训练权重时使用)"""
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
               512, 512, 512, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=False))  # 禁用 inplace
                in_channels = v
        
        return layers
    
    def _disable_inplace_relu(self, layers: list) -> list:
        """将 ReLU 的 inplace 设置为 False，避免梯度计算冲突"""
        new_layers = []
        for layer in layers:
            if isinstance(layer, nn.ReLU):
                new_layers.append(nn.ReLU(inplace=False))
            else:
                new_layers.append(layer)
        return new_layers
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        target_size = (h, w)
        
        # Forward through stages
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        
        # Side outputs
        side1 = self.side1(s1, target_size)
        side2 = self.side2(s2, target_size)
        side3 = self.side3(s3, target_size)
        side4 = self.side4(s4, target_size)
        side5 = self.side5(s5, target_size)
        
        # Fusion
        fused = self.fuse(torch.cat([side1, side2, side3, side4, side5], dim=1))
        
        # 应用 sigmoid
        outputs = {
            'side1': torch.sigmoid(side1),
            'side2': torch.sigmoid(side2),
            'side3': torch.sigmoid(side3),
            'side4': torch.sigmoid(side4),
            'side5': torch.sigmoid(side5),
            'fuse': torch.sigmoid(fused)
        }
        
        return outputs


# ============================================================================
# RCF: Richer Convolutional Features
# ============================================================================
class RCF(nn.Module):
    """
    RCF: Richer Convolutional Features for Edge Detection
    
    论文: Richer Convolutional Features for Edge Detection (CVPR 2017)
    改进的 HED，使用更丰富的卷积特征
    
    Args:
        pretrained: 是否使用预训练的 VGG16 权重
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # VGG16 特征提取器
        if HAS_TORCHVISION and pretrained:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            features = list(vgg.features.children())
            # 将所有 ReLU 的 inplace 设置为 False，避免梯度计算冲突
            features = self._disable_inplace_relu(features)
        else:
            features = self._make_vgg_features()
        
        # Stage 1-5 (与 HED 相同)
        self.stage1 = nn.Sequential(*features[:4])
        self.stage2 = nn.Sequential(*features[4:9])
        self.stage3 = nn.Sequential(*features[9:16])
        self.stage4 = nn.Sequential(*features[16:23])
        self.stage5 = nn.Sequential(*features[23:30])
        
        # RCF 特有：每个 conv 层都有输出
        # Stage 1: 2个conv (64 channels each)
        self.side1_1 = nn.Conv2d(64, 21, kernel_size=1)
        self.side1_2 = nn.Conv2d(64, 21, kernel_size=1)
        
        # Stage 2: 2个conv (128 channels each)
        self.side2_1 = nn.Conv2d(128, 21, kernel_size=1)
        self.side2_2 = nn.Conv2d(128, 21, kernel_size=1)
        
        # Stage 3: 3个conv (256 channels each)
        self.side3_1 = nn.Conv2d(256, 21, kernel_size=1)
        self.side3_2 = nn.Conv2d(256, 21, kernel_size=1)
        self.side3_3 = nn.Conv2d(256, 21, kernel_size=1)
        
        # Stage 4: 3个conv (512 channels each)
        self.side4_1 = nn.Conv2d(512, 21, kernel_size=1)
        self.side4_2 = nn.Conv2d(512, 21, kernel_size=1)
        self.side4_3 = nn.Conv2d(512, 21, kernel_size=1)
        
        # Stage 5: 3个conv (512 channels each)
        self.side5_1 = nn.Conv2d(512, 21, kernel_size=1)
        self.side5_2 = nn.Conv2d(512, 21, kernel_size=1)
        self.side5_3 = nn.Conv2d(512, 21, kernel_size=1)
        
        # 每个 stage 的融合
        self.fuse1 = nn.Conv2d(21 * 2, 1, kernel_size=1)
        self.fuse2 = nn.Conv2d(21 * 2, 1, kernel_size=1)
        self.fuse3 = nn.Conv2d(21 * 3, 1, kernel_size=1)
        self.fuse4 = nn.Conv2d(21 * 3, 1, kernel_size=1)
        self.fuse5 = nn.Conv2d(21 * 3, 1, kernel_size=1)
        
        # 最终融合
        self.final_fuse = nn.Conv2d(5, 1, kernel_size=1)
        
        self._init_weights()
    
    def _make_vgg_features(self) -> list:
        """创建 VGG 特征提取层"""
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
               512, 512, 512, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=False))  # 禁用 inplace
                in_channels = v
        
        return layers

    def _disable_inplace_relu(self, layers: list) -> list:
        """将 ReLU 的 inplace 设置为 False，避免梯度计算冲突"""
        new_layers = []
        for layer in layers:
            if isinstance(layer, nn.ReLU):
                new_layers.append(nn.ReLU(inplace=False))
            else:
                new_layers.append(layer)
        return new_layers
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def _upsample(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        target_size = (h, w)
        
        # Stage 1
        x = self.stage1[0](x)  # conv1_1
        s1_1 = self.side1_1(x)
        x = self.stage1[1](x)  # relu
        x = self.stage1[2](x)  # conv1_2
        s1_2 = self.side1_2(x)
        x = self.stage1[3](x)  # relu, pool
        
        fuse1 = self.fuse1(torch.cat([s1_1, s1_2], dim=1))
        fuse1 = self._upsample(fuse1, target_size)
        
        # Stage 2
        x = self.stage2[0](x)  # pool2 (channels remain 64)
        x = self.stage2[1](x)  # conv2_1 -> 128 ch
        s2_1 = self.side2_1(x)
        x = self.stage2[2](x)  # relu
        x = self.stage2[3](x)  # conv2_2 -> 128 ch
        s2_2 = self.side2_2(x)
        x = self.stage2[4](x)  # relu
        
        fuse2 = self.fuse2(torch.cat([s2_1, s2_2], dim=1))
        fuse2 = self._upsample(fuse2, target_size)
        
        # Stage 3
        x = self.stage3[0](x)  # pool3
        x = self.stage3[1](x)  # conv3_1 -> 256 ch
        s3_1 = self.side3_1(x)
        x = self.stage3[2](x)  # relu
        x = self.stage3[3](x)  # conv3_2 -> 256 ch
        s3_2 = self.side3_2(x)
        x = self.stage3[4](x)  # relu
        x = self.stage3[5](x)  # conv3_3 -> 256 ch
        s3_3 = self.side3_3(x)
        x = self.stage3[6](x)  # relu
        
        fuse3 = self.fuse3(torch.cat([s3_1, s3_2, s3_3], dim=1))
        fuse3 = self._upsample(fuse3, target_size)
        
        # Stage 4
        x = self.stage4[0](x)  # pool4
        x = self.stage4[1](x)  # conv4_1 -> 512 ch
        s4_1 = self.side4_1(x)
        x = self.stage4[2](x)  # relu
        x = self.stage4[3](x)  # conv4_2 -> 512 ch
        s4_2 = self.side4_2(x)
        x = self.stage4[4](x)  # relu
        x = self.stage4[5](x)  # conv4_3 -> 512 ch
        s4_3 = self.side4_3(x)
        x = self.stage4[6](x)  # relu
        
        fuse4 = self.fuse4(torch.cat([s4_1, s4_2, s4_3], dim=1))
        fuse4 = self._upsample(fuse4, target_size)
        
        # Stage 5
        x = self.stage5[0](x)  # pool5
        x = self.stage5[1](x)  # conv5_1 -> 512 ch
        s5_1 = self.side5_1(x)
        x = self.stage5[2](x)  # relu
        x = self.stage5[3](x)  # conv5_2 -> 512 ch
        s5_2 = self.side5_2(x)
        x = self.stage5[4](x)  # relu
        x = self.stage5[5](x)  # conv5_3 -> 512 ch
        s5_3 = self.side5_3(x)
        x = self.stage5[6](x)  # relu
        
        fuse5 = self.fuse5(torch.cat([s5_1, s5_2, s5_3], dim=1))
        fuse5 = self._upsample(fuse5, target_size)
        
        # Final fusion
        final = self.final_fuse(torch.cat([fuse1, fuse2, fuse3, fuse4, fuse5], dim=1))
        
        outputs = {
            'side1': torch.sigmoid(fuse1),
            'side2': torch.sigmoid(fuse2),
            'side3': torch.sigmoid(fuse3),
            'side4': torch.sigmoid(fuse4),
            'side5': torch.sigmoid(fuse5),
            'fuse': torch.sigmoid(final)
        }
        
        return outputs


# ============================================================================
# BDCN: Bi-Directional Cascade Network
# ============================================================================
class BDCN(nn.Module):
    """
    BDCN: Bi-Directional Cascade Network for Perceptual Edge Detection
    
    论文: Bi-Directional Cascade Network for Perceptual Edge Detection (CVPR 2019)
    
    简化实现：保持核心的双向级联结构
    
    Args:
        pretrained: 是否使用预训练权重
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Encoder (VGG-like)
        self.conv1 = nn.Sequential(
            ConvBnRelu(3, 64),
            ConvBnRelu(64, 64)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Sequential(
            ConvBnRelu(64, 128),
            ConvBnRelu(128, 128)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Sequential(
            ConvBnRelu(128, 256),
            ConvBnRelu(256, 256),
            ConvBnRelu(256, 256)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Sequential(
            ConvBnRelu(256, 512),
            ConvBnRelu(512, 512),
            ConvBnRelu(512, 512)
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Sequential(
            ConvBnRelu(512, 512),
            ConvBnRelu(512, 512),
            ConvBnRelu(512, 512)
        )
        
        # Scale-specific edge detectors (Bi-directional)
        self.dsn1 = self._make_dsn(64, 1)
        self.dsn2 = self._make_dsn(128, 2)
        self.dsn3 = self._make_dsn(256, 4)
        self.dsn4 = self._make_dsn(512, 8)
        self.dsn5 = self._make_dsn(512, 16)
        
        # Bi-directional cascade connections
        # Forward direction
        self.cascade_f1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.cascade_f2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.cascade_f3 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.cascade_f4 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        
        # Backward direction
        self.cascade_b5 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.cascade_b4 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.cascade_b3 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.cascade_b2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        
        # Final fusion
        self.fuse = nn.Conv2d(10, 1, kernel_size=1)
        
        if pretrained and HAS_TORCHVISION:
            self._load_pretrained_vgg()
    
    def _make_dsn(self, in_channels: int, scale: int) -> nn.Sequential:
        """创建 Deep Supervision Network 模块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )
    
    def _load_pretrained_vgg(self):
        """加载预训练 VGG 权重"""
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        vgg_features = list(vgg.features.children())
        
        # 复制权重到对应层
        self.conv1[0].conv.weight.data = vgg_features[0].weight.data
        self.conv1[1].conv.weight.data = vgg_features[2].weight.data
        self.conv2[0].conv.weight.data = vgg_features[5].weight.data
        self.conv2[1].conv.weight.data = vgg_features[7].weight.data
    
    def _upsample(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        target_size = (h, w)
        
        # Encoder
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv5(self.pool4(c4))
        
        # DSN outputs
        d1 = self.dsn1(c1)
        d2 = self.dsn2(c2)
        d3 = self.dsn3(c3)
        d4 = self.dsn4(c4)
        d5 = self.dsn5(c5)
        
        # Upsample all to original size
        d1 = self._upsample(d1, target_size)
        d2 = self._upsample(d2, target_size)
        d3 = self._upsample(d3, target_size)
        d4 = self._upsample(d4, target_size)
        d5 = self._upsample(d5, target_size)
        
        # Forward cascade
        f1 = d1
        f2 = d2 + self.cascade_f1(f1)
        f3 = d3 + self.cascade_f2(f2)
        f4 = d4 + self.cascade_f3(f3)
        f5 = d5 + self.cascade_f4(f4)
        
        # Backward cascade
        b5 = d5
        b4 = d4 + self.cascade_b5(b5)
        b3 = d3 + self.cascade_b4(b4)
        b2 = d2 + self.cascade_b3(b3)
        b1 = d1 + self.cascade_b2(b2)
        
        # Fusion
        all_outputs = torch.cat([f1, f2, f3, f4, f5, b1, b2, b3, b4, b5], dim=1)
        fused = self.fuse(all_outputs)
        
        outputs = {
            'side1': torch.sigmoid(d1),
            'side2': torch.sigmoid(d2),
            'side3': torch.sigmoid(d3),
            'side4': torch.sigmoid(d4),
            'side5': torch.sigmoid(d5),
            'forward': torch.sigmoid(f5),
            'backward': torch.sigmoid(b1),
            'fuse': torch.sigmoid(fused)
        }
        
        return outputs


# ============================================================================
# DexiNed: Dense Extreme Inception Network
# ============================================================================
class DexiBlock(nn.Module):
    """DexiNed 的 Dense Extreme Inception Block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        mid_channels = out_channels // 4
        
        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )
        
        # Branch 2: 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )
        
        # Branch 3: 5x5 conv (用两个3x3近似)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )
        
        # Branch 4: Dilated conv
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )
        
        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fuse(out)
        
        return out


class DexiNed(nn.Module):
    """
    DexiNed: Dense Extreme Inception Network for Edge Detection
    
    论文: Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection (WACV 2020)
    
    Args:
        in_channels: 输入通道数
    """
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Initial conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        
        # Dense Extreme Inception Blocks
        self.block1 = DexiBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.block2 = DexiBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.block3 = DexiBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.block4 = DexiBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.block5 = DexiBlock(512, 512)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.block6 = DexiBlock(512, 512)
        
        # Side outputs
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(128, 1, kernel_size=1)
        self.side3 = nn.Conv2d(256, 1, kernel_size=1)
        self.side4 = nn.Conv2d(512, 1, kernel_size=1)
        self.side5 = nn.Conv2d(512, 1, kernel_size=1)
        self.side6 = nn.Conv2d(512, 1, kernel_size=1)
        
        # Final fusion
        self.fuse = nn.Conv2d(6, 1, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _upsample(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        target_size = (h, w)
        
        # Encoder
        x = self.init_conv(x)
        
        b1 = self.block1(x)
        b2 = self.block2(self.pool1(b1))
        b3 = self.block3(self.pool2(b2))
        b4 = self.block4(self.pool3(b3))
        b5 = self.block5(self.pool4(b4))
        b6 = self.block6(self.pool5(b5))
        
        # Side outputs
        s1 = self._upsample(self.side1(b1), target_size)
        s2 = self._upsample(self.side2(b2), target_size)
        s3 = self._upsample(self.side3(b3), target_size)
        s4 = self._upsample(self.side4(b4), target_size)
        s5 = self._upsample(self.side5(b5), target_size)
        s6 = self._upsample(self.side6(b6), target_size)
        
        # Fusion
        fused = self.fuse(torch.cat([s1, s2, s3, s4, s5, s6], dim=1))
        
        outputs = {
            'side1': torch.sigmoid(s1),
            'side2': torch.sigmoid(s2),
            'side3': torch.sigmoid(s3),
            'side4': torch.sigmoid(s4),
            'side5': torch.sigmoid(s5),
            'side6': torch.sigmoid(s6),
            'fuse': torch.sigmoid(fused)
        }
        
        return outputs


# ============================================================================
# 模型工厂函数
# ============================================================================
def get_edge_model(
    model_name: str,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    获取边缘检测模型的统一接口
    
    Args:
        model_name: 模型名称 ('hed', 'rcf', 'bdcn', 'dexined')
        pretrained: 是否使用预训练权重 (VGG backbone)
        **kwargs: 其他模型特定参数
    
    Returns:
        nn.Module: 边缘检测模型实例
    
    Note:
        所有模型返回一个字典，包含多个侧输出和融合输出。
        主要输出通常是 outputs['fuse']。
    
    Example:
        >>> model = get_edge_model('hed')
        >>> outputs = model(torch.randn(1, 3, 512, 512))
        >>> edge_map = outputs['fuse']  # 主要输出
        >>> print(edge_map.shape)  # torch.Size([1, 1, 512, 512])
    """
    model_name = model_name.lower().replace('-', '').replace('_', '')
    
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {SUPPORTED_MODELS}"
        )
    
    if model_name == 'hed':
        model = HED(pretrained=pretrained)
    
    elif model_name == 'rcf':
        model = RCF(pretrained=pretrained)
    
    elif model_name == 'bdcn':
        model = BDCN(pretrained=pretrained)
    
    elif model_name == 'dexined':
        model = DexiNed(**kwargs)
    
    else:
        raise ValueError(f"Model {model_name} is not implemented")
    
    return model


def get_model_info(model_name: str) -> Dict[str, Any]:
    """获取模型信息"""
    info = {
        'hed': {
            'name': 'HED',
            'full_name': 'Holistically-nested Edge Detection',
            'paper': 'Holistically-Nested Edge Detection',
            'venue': 'ICCV 2015',
            'backbone': 'VGG16'
        },
        'rcf': {
            'name': 'RCF',
            'full_name': 'Richer Convolutional Features',
            'paper': 'Richer Convolutional Features for Edge Detection',
            'venue': 'CVPR 2017',
            'backbone': 'VGG16'
        },
        'bdcn': {
            'name': 'BDCN',
            'full_name': 'Bi-Directional Cascade Network',
            'paper': 'Bi-Directional Cascade Network for Perceptual Edge Detection',
            'venue': 'CVPR 2019',
            'backbone': 'VGG16'
        },
        'dexined': {
            'name': 'DexiNed',
            'full_name': 'Dense Extreme Inception Network',
            'paper': 'Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection',
            'venue': 'WACV 2020',
            'backbone': 'Custom (Inception-based)'
        }
    }
    
    model_name = model_name.lower().replace('-', '').replace('_', '')
    return info.get(model_name, {})


def list_available_models() -> List[str]:
    """列出所有可用的边缘检测模型"""
    return SUPPORTED_MODELS.copy()


class EdgeModelWrapper(nn.Module):
    """
    边缘检测模型包装器
    
    将字典输出转换为单一张量输出，便于训练
    """
    
    def __init__(self, model: nn.Module, output_key: str = 'fuse'):
        super().__init__()
        self.model = model
        self.output_key = output_key
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs[self.output_key]
    
    def forward_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """返回所有输出"""
        return self.model(x)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Edge Detection Model Factory Test")
    print("=" * 60)
    print(f"torchvision available: {HAS_TORCHVISION}")
    print(f"Available models: {list_available_models()}")
    print()
    
    # 测试输入
    x = torch.randn(2, 3, 256, 256)
    
    for model_name in SUPPORTED_MODELS:
        try:
            print(f"Testing {model_name}...")
            info = get_model_info(model_name)
            print(f"  Full name: {info.get('full_name', 'N/A')}")
            print(f"  Venue: {info.get('venue', 'N/A')}")
            
            model = get_edge_model(model_name, pretrained=False)
            model.eval()
            
            with torch.no_grad():
                outputs = model(x)
            
            # 计算参数量
            params = sum(p.numel() for p in model.parameters()) / 1e6
            
            print(f"  ✓ Output keys: {list(outputs.keys())}")
            print(f"  ✓ Fuse output shape: {outputs['fuse'].shape}")
            print(f"  ✓ Parameters: {params:.2f}M")
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
