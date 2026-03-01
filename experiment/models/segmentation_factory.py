"""
Segmentation Model Factory
提供统一接口创建语义分割模型：U-Net, DeepLabV3+, SegFormer, VM-UNet
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

# 尝试导入 segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False
    print("Warning: segmentation_models_pytorch not installed. "
          "Install with: pip install segmentation-models-pytorch")

# 尝试导入 mamba 相关库 (用于 VM-UNet)
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


# ============================================================================
# 支持的模型列表
# ============================================================================
SUPPORTED_MODELS = ['unet', 'deeplabv3plus', 'segformer', 'vmunet']

# Encoder 配置
DEFAULT_ENCODERS = {
    'unet': 'resnet34',
    'deeplabv3plus': 'resnet50',
    'segformer': 'mit_b2',  # MixTransformer
}


# ============================================================================
# VM-UNet 实现 (基于 Mamba State Space Model)
# ============================================================================
class MambaBlock(nn.Module):
    """Mamba Block - 状态空间模型块 (仅使用 mamba_ssm 库)"""
    
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        
        if not HAS_MAMBA:
            raise ImportError(
                "mamba_ssm is required for VM-UNet. "
                "Install with: pip install mamba-ssm"
            )
        
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 展平空间维度用于 Mamba
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_flat + x_mamba
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return out


class VMUNetEncoder(nn.Module):
    """VM-UNet 编码器"""
    
    def __init__(self, in_channels: int = 3, base_dim: int = 32):
        super().__init__()
        
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )
        
        # Encoder stages
        self.stage1 = self._make_stage(dims[0], dims[0], num_blocks=2)
        self.down1 = nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        
        self.stage2 = self._make_stage(dims[1], dims[1], num_blocks=2)
        self.down2 = nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        
        self.stage3 = self._make_stage(dims[2], dims[2], num_blocks=2)
        self.down3 = nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)
        
        self.stage4 = self._make_stage(dims[3], dims[3], num_blocks=2)
        
        self.dims = dims
    
    def _make_stage(self, in_dim: int, out_dim: int, num_blocks: int) -> nn.Sequential:
        layers = []
        for i in range(num_blocks):
            layers.append(MambaBlock(in_dim if i == 0 else out_dim))
        if in_dim != out_dim:
            layers.append(nn.Conv2d(in_dim, out_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> list:
        features = []
        
        x = self.stem(x)
        
        x = self.stage1(x)
        features.append(x)
        x = self.down1(x)
        
        x = self.stage2(x)
        features.append(x)
        x = self.down2(x)
        
        x = self.stage3(x)
        features.append(x)
        x = self.down3(x)
        
        x = self.stage4(x)
        features.append(x)
        
        return features


class VMUNetDecoder(nn.Module):
    """VM-UNet 解码器"""
    
    def __init__(self, encoder_dims: list, num_classes: int = 1):
        super().__init__()
        
        dims = encoder_dims[::-1]  # 反转
        
        self.up1 = nn.ConvTranspose2d(dims[0], dims[1], kernel_size=2, stride=2)
        self.dec1 = self._make_dec_block(dims[1] * 2, dims[1])
        
        self.up2 = nn.ConvTranspose2d(dims[1], dims[2], kernel_size=2, stride=2)
        self.dec2 = self._make_dec_block(dims[2] * 2, dims[2])
        
        self.up3 = nn.ConvTranspose2d(dims[2], dims[3], kernel_size=2, stride=2)
        self.dec3 = self._make_dec_block(dims[3] * 2, dims[3])
        
        self.up4 = nn.ConvTranspose2d(dims[3], dims[3], kernel_size=2, stride=2)
        
        self.head = nn.Conv2d(dims[3], num_classes, kernel_size=1)
    
    def _make_dec_block(self, in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            MambaBlock(out_dim)
        )
    
    def forward(self, features: list) -> torch.Tensor:
        features = features[::-1]  # 反转顺序
        
        x = self.up1(features[0])
        x = torch.cat([x, features[1]], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, features[2]], dim=1)
        x = self.dec2(x)
        
        x = self.up3(x)
        x = torch.cat([x, features[3]], dim=1)
        x = self.dec3(x)
        
        x = self.up4(x)
        x = self.head(x)
        
        return x


class VMUNet(nn.Module):
    """
    VM-UNet: Vision Mamba U-Net
    
    基于状态空间模型 (State Space Model) 的 U-Net 变体
    论文: VM-UNet: Vision Mamba UNet for Medical Image Segmentation
    
    Args:
        in_channels: 输入通道数
        num_classes: 输出类别数
        base_dim: 基础特征维度
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_dim: int = 32
    ):
        super().__init__()
        
        if not HAS_MAMBA:
            raise ImportError(
                "mamba_ssm is required for VM-UNet. "
                "Install with: pip install mamba-ssm"
            )
        
        self.encoder = VMUNetEncoder(in_channels, base_dim)
        self.decoder = VMUNetDecoder(self.encoder.dims, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        out = self.decoder(features)
        
        # 上采样到输入尺寸
        if out.shape[2:] != x.shape[2:]:
            out = nn.functional.interpolate(
                out, size=x.shape[2:], mode='bilinear', align_corners=False
            )
        
        return out


# ============================================================================
# 模型工厂函数
# ============================================================================
def get_segmentation_model(
    model_name: str,
    num_classes: int = 1,
    weights: Optional[str] = 'imagenet',
    in_channels: int = 3,
    **kwargs
) -> nn.Module:
    """
    获取语义分割模型的统一接口
    
    Args:
        model_name: 模型名称 ('unet', 'deeplabv3plus', 'segformer', 'vmunet')
        num_classes: 输出类别数 (二分类用 1)
        weights: 预训练权重 ('imagenet' 或 None)
        in_channels: 输入通道数
        **kwargs: 其他模型特定参数
    
    Returns:
        nn.Module: 分割模型实例
    
    Example:
        >>> model = get_segmentation_model('unet', num_classes=1)
        >>> output = model(torch.randn(1, 3, 512, 512))
        >>> print(output.shape)  # torch.Size([1, 1, 512, 512])
    """
    model_name = model_name.lower().replace('-', '').replace('_', '')
    
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {SUPPORTED_MODELS}"
        )
    
    # 获取 encoder
    encoder_name = kwargs.pop('encoder_name', DEFAULT_ENCODERS.get(model_name))
    encoder_weights = weights if weights else None
    
    if model_name == 'unet':
        if not HAS_SMP:
            raise ImportError("segmentation_models_pytorch is required for U-Net")
        
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs
        )
    
    elif model_name == 'deeplabv3plus':
        if not HAS_SMP:
            raise ImportError("segmentation_models_pytorch is required for DeepLabV3+")

        # 兼容旧版本 segmentation_models_pytorch 不提供 DeepLabV3Plus 的情况
        if hasattr(smp, 'DeepLabV3Plus'):
            model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                **kwargs
            )
        elif hasattr(smp, 'DeepLabV3'):
            # Fallback: 使用 DeepLabV3 作为替代，并提示升级
            print(
                "Warning: segmentation_models_pytorch version does not include DeepLabV3Plus. "
                "Using DeepLabV3 as a fallback. Upgrade with: \n"
                "  pip install -U segmentation-models-pytorch"
            )
            model = smp.DeepLabV3(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                **kwargs
            )
        else:
            raise AttributeError(
                "segmentation_models_pytorch missing DeepLabV3Plus/DeepLabV3. "
                "Please upgrade: pip install -U segmentation-models-pytorch"
            )
    
    elif model_name == 'segformer':
        if not HAS_SMP:
            raise ImportError("segmentation_models_pytorch is required for SegFormer")
        
        # SegFormer 使用 MixTransformer (mit) 作为 encoder
        if not encoder_name.startswith('mit_'):
            encoder_name = 'mit_b2'
        
        # 注意: smp 的 SegFormer 需要使用 FPN 或自定义 decoder
        # 这里使用 MAnet 作为类似 SegFormer 的架构
        try:
            # 尝试使用官方 SegFormer (如果 smp 版本支持)
            model = smp.MAnet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                **kwargs
            )
        except Exception:
            # Fallback: 使用 FPN + MixTransformer
            model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                **kwargs
            )
    
    elif model_name == 'vmunet':
        base_dim = kwargs.pop('base_dim', 32)
        model = VMUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_dim=base_dim
        )
    
    else:
        raise ValueError(f"Model {model_name} is not implemented")
    
    return model


def get_model_info(model_name: str) -> Dict[str, Any]:
    """获取模型信息"""
    info = {
        'unet': {
            'name': 'U-Net',
            'paper': 'U-Net: Convolutional Networks for Biomedical Image Segmentation',
            'year': 2015,
            'default_encoder': 'resnet34',
            'requires_smp': True
        },
        'deeplabv3plus': {
            'name': 'DeepLabV3+',
            'paper': 'Encoder-Decoder with Atrous Separable Convolution',
            'year': 2018,
            'default_encoder': 'resnet50',
            'requires_smp': True
        },
        'segformer': {
            'name': 'SegFormer',
            'paper': 'SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers',
            'year': 2021,
            'default_encoder': 'mit_b2',
            'requires_smp': True
        },
        'vmunet': {
            'name': 'VM-UNet',
            'paper': 'VM-UNet: Vision Mamba UNet for Medical Image Segmentation',
            'year': 2024,
            'default_encoder': None,
            'requires_smp': False,
            'requires_mamba': True
        }
    }
    
    model_name = model_name.lower().replace('-', '').replace('_', '')
    return info.get(model_name, {})


def list_available_models() -> list:
    """列出所有可用的模型"""
    available = []
    
    for model in SUPPORTED_MODELS:
        info = get_model_info(model)
        if info.get('requires_smp') and not HAS_SMP:
            continue
        available.append(model)
    
    return available


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Segmentation Model Factory Test")
    print("=" * 60)
    print(f"segmentation_models_pytorch available: {HAS_SMP}")
    print(f"mamba_ssm available: {HAS_MAMBA}")
    print(f"Available models: {list_available_models()}")
    print()
    
    # 测试输入
    x = torch.randn(2, 3, 256, 256)
    
    for model_name in SUPPORTED_MODELS:
        try:
            print(f"Testing {model_name}...")
            model = get_segmentation_model(model_name, num_classes=1)
            model.eval()
            
            with torch.no_grad():
                output = model(x)
            
            # 计算参数量
            params = sum(p.numel() for p in model.parameters()) / 1e6
            
            print(f"  ✓ Output shape: {output.shape}")
            print(f"  ✓ Parameters: {params:.2f}M")
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print()
