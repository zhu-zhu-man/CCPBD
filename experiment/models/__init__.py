"""
CCPBD Models Module
提供语义分割和边缘检测模型的统一接口
"""

from .segmentation_factory import (
    get_segmentation_model,
    get_model_info as get_segmentation_model_info,
    list_available_models as list_segmentation_models,
    SUPPORTED_MODELS as SEGMENTATION_MODELS,
    HAS_SMP,
    HAS_MAMBA,
    VMUNet
)

from .edge_factory import (
    get_edge_model,
    get_model_info as get_edge_model_info,
    list_available_models as list_edge_models,
    SUPPORTED_MODELS as EDGE_MODELS,
    EdgeModelWrapper,
    HED,
    RCF,
    BDCN,
    DexiNed
)

__all__ = [
    # Segmentation
    'get_segmentation_model',
    'get_segmentation_model_info',
    'list_segmentation_models',
    'SEGMENTATION_MODELS',
    'HAS_SMP',
    'HAS_MAMBA',
    'VMUNet',
    
    # Edge Detection
    'get_edge_model',
    'get_edge_model_info',
    'list_edge_models',
    'EDGE_MODELS',
    'EdgeModelWrapper',
    'HED',
    'RCF',
    'BDCN',
    'DexiNed'
]
