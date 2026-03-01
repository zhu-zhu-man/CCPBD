# creat_gaussian_distance_map.py 高斯距离标签生成
import numpy as np
from scipy.ndimage import distance_transform_edt

def generate_gaussian_distance_map(mask, sigma=20.0):
    """ 生成高斯距离图 """
    # 二值化：前景=1，背景=0
    binary = (mask > 0).astype(np.uint8)
    # 欧氏距离变换
    dist = distance_transform_edt(binary)
    # 高斯映射
    gaussian = np.exp(- (dist ** 2) / (2 * sigma ** 2))
    # 映射到 [0, 255]
    dist_map = (gaussian * 255).astype(np.uint8)
    return dist_map