"""
CCPBD Dataset - Chinese Cropland Parcel and Boundary Dataset
用于农田地块分割和边界检测的 PyTorch 数据集类
"""

from __future__ import annotations

import os
from typing import Tuple, Optional, Callable, List, Any, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
    # 轻量版本检测，兼容 Albumentations 2.x 的 API 变化（RandomResizedCrop 使用 size）
    try:
        _ALBU_MAJOR = int(getattr(A, "__version__", "1.0.0").split(".")[0])
    except Exception:
        _ALBU_MAJOR = 1
except ImportError:
    HAS_ALBUMENTATIONS = False
    A = None  # type: ignore
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF

if TYPE_CHECKING:  # 为类型检查提供 A 符号，运行时不强依赖
    import albumentations as A


# ImageNet 标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CCPBDDataset(Dataset):
    """
    CCPBD (Chinese Cropland Parcel and Boundary Dataset) 数据集类
    
    支持两种任务：
    - segmentation: 农田地块语义分割
    - edge: 农田边界检测
    
    Args:
        root_dir: 数据集根目录
        split: 数据划分 ('train', 'val', 'test')
        task: 任务类型 ('segmentation' 或 'edge')
        image_size: 图像尺寸 (默认 512)
        transform: 自定义变换 (可选，如果提供则覆盖默认变换)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        task: str = 'segmentation',
        image_size: int = 512,
        transform: Optional[Callable] = None
    ):
        assert split in ['train', 'val', 'test'], \
            f"split must be 'train', 'val', or 'test', got {split}"
        assert task in ['segmentation', 'edge'], \
            f"task must be 'segmentation' or 'edge', got {task}"
        
        self.root_dir = root_dir
        self.split = split
        self.task = task
        self.image_size = image_size
        
        # 设置路径 - 适配 CCPBD 数据集结构: root/split/image, root/split/mask, root/split/boundary
        self.split_dir = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.split_dir, 'image')
        self.mask_dir = os.path.join(self.split_dir, 'mask')
        self.edge_dir = os.path.join(self.split_dir, 'boundary')
        
        # 读取文件列表
        self.file_ids = self._load_file_ids()
        
        # 设置数据变换
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()
    
    def _load_file_ids(self) -> List[str]:
        """从目录自动扫描文件 ID 列表（支持 PNG 格式）"""
        # 首先尝试从 txt 文件读取
        split_file = os.path.join(self.root_dir, f'{self.split}.txt')
        
        if os.path.exists(split_file):
            with open(split_file, 'r', encoding='utf-8') as f:
                file_ids = [line.strip() for line in f.readlines() if line.strip()]
            if len(file_ids) > 0:
                return file_ids
        
        # 如果没有 txt 文件，则从图像目录自动扫描
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # 扫描图像目录中的 PNG 文件
        file_ids = []
        for filename in sorted(os.listdir(self.image_dir)):
            if filename.lower().endswith('.png'):
                # 去除扩展名作为文件 ID
                file_id = os.path.splitext(filename)[0]
                file_ids.append(file_id)
        
        if len(file_ids) == 0:
            raise ValueError(f"No PNG files found in {self.image_dir}")
        
        return file_ids
    
    def _get_default_transform(self) -> Callable:
        """获取默认的数据变换"""
        if HAS_ALBUMENTATIONS:
            return self._get_albumentations_transform()
        else:
            return self._get_torchvision_transform()
    
    def _get_albumentations_transform(self) -> Any:
        """使用 albumentations 创建数据变换"""
        if self.split == 'train':
            # 训练集：数据增强
            rrc = (
                A.RandomResizedCrop(
                    size=(self.image_size, self.image_size),
                    scale=(0.5, 1.0),
                    ratio=(0.75, 1.33),
                    p=1.0
                ) if HAS_ALBUMENTATIONS and _ALBU_MAJOR >= 2 else
                A.RandomResizedCrop(
                    height=self.image_size,
                    width=self.image_size,
                    scale=(0.5, 1.0),
                    ratio=(0.75, 1.33),
                    p=1.0
                )
            )
            transform = A.Compose([
                rrc,
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ])
        else:
            # 验证集/测试集：只做 resize 和标准化
            # Resize 在 1.x 与 2.x 中 API 一致，这里保持原写法
            transform = A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2()
            ])
        
        return transform
    
    def _get_torchvision_transform(self) -> Callable:
        """使用 torchvision 创建数据变换 (备选方案)"""
        if self.split == 'train':
            return TorchvisionTrainTransform(
                self.image_size,
                IMAGENET_MEAN,
                IMAGENET_STD
            )
        else:
            return TorchvisionValTransform(
                self.image_size,
                IMAGENET_MEAN,
                IMAGENET_STD
            )
    
    def _load_image(self, file_id: str) -> np.ndarray:
        """加载 RGB 图像"""
        # 优先使用 PNG 格式，然后尝试其他格式
        extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
        
        for ext in extensions:
            image_path = os.path.join(self.image_dir, f'{file_id}{ext}')
            if os.path.exists(image_path):
                break
        else:
            raise FileNotFoundError(
                f"Image not found for {file_id} in {self.image_dir}"
            )
        
        # 使用 OpenCV 读取图像 (BGR -> RGB)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_mask(self, file_id: str) -> np.ndarray:
        """加载分割掩膜或边缘图"""
        if self.task == 'segmentation':
            label_dir = self.mask_dir
        else:  # edge
            label_dir = self.edge_dir
        
        # 尝试多种可能的标签扩展名
        extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
        
        for ext in extensions:
            mask_path = os.path.join(label_dir, f'{file_id}{ext}')
            if os.path.exists(mask_path):
                break
        else:
            # 尝试带后缀的文件名
            suffixes = ['_mask', '_edge', '_label']
            found = False
            for suffix in suffixes:
                for ext in extensions:
                    mask_path = os.path.join(label_dir, f'{file_id}{suffix}{ext}')
                    if os.path.exists(mask_path):
                        found = True
                        break
                if found:
                    break
            else:
                raise FileNotFoundError(
                    f"Mask not found for {file_id} in {label_dir}"
                )
        
        # 读取灰度图
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # 确保二值化 (0 和 1)
        mask = (mask > 0).astype(np.uint8)
        
        return mask
    
    def __len__(self) -> int:
        return len(self.file_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        获取单个样本
        
        Returns:
            image: 形状为 (3, H, W) 的图像张量，已标准化
            mask: 形状为 (1, H, W) 的掩膜张量，值为 0 或 1
            filename: 文件 ID (用于保存结果)
        """
        file_id = self.file_ids[idx]
        
        # 加载图像和掩膜
        image = self._load_image(file_id)
        mask = self._load_mask(file_id)
        
        # 应用变换
        if HAS_ALBUMENTATIONS and isinstance(self.transform, A.Compose):
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image, mask = self.transform(image, mask)
        
        # 确保 mask 是 (1, H, W) 形状的 float tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        mask = mask.float()
        
        return image, mask, file_id


class TorchvisionTrainTransform:
    """使用 torchvision 的训练数据变换 (备选方案)"""
    
    def __init__(self, image_size: int, mean: List[float], std: List[float]):
        self.image_size = image_size
        self.mean = mean
        self.std = std
    
    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 转换为 PIL Image
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        
        # 随机 resize crop
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=(0.5, 1.0), ratio=(0.75, 1.33)
        )
        image = TF.resized_crop(
            image, i, j, h, w,
            (self.image_size, self.image_size),
            interpolation=T.InterpolationMode.BILINEAR
        )
        mask = TF.resized_crop(
            mask, i, j, h, w,
            (self.image_size, self.image_size),
            interpolation=T.InterpolationMode.NEAREST
        )
        
        # 随机水平翻转
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # 随机垂直翻转
        if torch.rand(1) > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # 转换为张量并标准化
        image = TF.to_tensor(image)  # [0, 1]
        image = TF.normalize(image, self.mean, self.std)
        
        mask = torch.from_numpy(np.array(mask)).long()
        mask = (mask > 0).float()
        
        return image, mask


class TorchvisionValTransform:
    """使用 torchvision 的验证数据变换 (备选方案)"""
    
    def __init__(self, image_size: int, mean: List[float], std: List[float]):
        self.image_size = image_size
        self.mean = mean
        self.std = std
    
    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 转换为 PIL Image
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        
        # Resize
        image = TF.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=T.InterpolationMode.BILINEAR
        )
        mask = TF.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=T.InterpolationMode.NEAREST
        )
        
        # 转换为张量并标准化
        image = TF.to_tensor(image)  # [0, 1]
        image = TF.normalize(image, self.mean, self.std)
        
        mask = torch.from_numpy(np.array(mask)).long()
        mask = (mask > 0).float()
        
        return image, mask


def get_dataloader(
    root_dir: str,
    split: str,
    task: str = 'segmentation',
    image_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> torch.utils.data.DataLoader:
    """
    获取数据加载器的便捷函数
    
    Args:
        root_dir: 数据集根目录
        split: 数据划分
        task: 任务类型
        image_size: 图像尺寸
        batch_size: 批量大小
        num_workers: 数据加载线程数
        pin_memory: 是否锁页内存
        drop_last: 是否丢弃最后不完整的批次
    
    Returns:
        DataLoader 实例
    """
    dataset = CCPBDDataset(
        root_dir=root_dir,
        split=split,
        task=task,
        image_size=image_size
    )
    
    shuffle = (split == 'train')
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last and (split == 'train')
    )
    
    return dataloader


if __name__ == '__main__':
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CCPBD Dataset')
    parser.add_argument('--root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--task', type=str, default='segmentation', choices=['segmentation', 'edge'])
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()
    
    print(f"Testing CCPBDDataset...")
    print(f"  Root: {args.root}")
    print(f"  Split: {args.split}")
    print(f"  Task: {args.task}")
    print(f"  Image size: {args.image_size}")
    print(f"  Using albumentations: {HAS_ALBUMENTATIONS}")
    
    try:
        dataset = CCPBDDataset(
            root_dir=args.root,
            split=args.split,
            task=args.task,
            image_size=args.image_size
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"  Number of samples: {len(dataset)}")
        
        # 测试加载第一个样本
        image, mask, filename = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Filename: {filename}")
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask dtype: {mask.dtype}")
        print(f"  Mask unique values: {torch.unique(mask).tolist()}")
        
        # 测试 DataLoader
        dataloader = get_dataloader(
            root_dir=args.root,
            split=args.split,
            task=args.task,
            image_size=args.image_size,
            batch_size=2,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        images, masks, filenames = batch
        print(f"\nBatch test:")
        print(f"  Batch images shape: {images.shape}")
        print(f"  Batch masks shape: {masks.shape}")
        print(f"  Filenames: {filenames}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
