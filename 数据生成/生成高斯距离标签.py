import numpy as np
import os
import cv2

def read_png(filename):
    """读取PNG图像，返回 shape=(3, H, W) 格式的数组"""
    img = cv2.imread(filename, cv2.IMREAD_COLOR)  # shape=(H, W, 3), BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # 转为RGB
    img = np.transpose(img, (2, 0, 1))            # 转为 (3, H, W)
    return img

def write_png(filename, im_data):
    """写入PNG图像，输入为 shape=(3, H, W)"""
    img = np.transpose(im_data, (1, 2, 0))  # 转为 (H, W, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

# 路径设置
maskRoot = r"G:\data\newlabels"
distRoot = r"G:\data\dist"


os.makedirs(distRoot, exist_ok=True)

for imgName in os.listdir(maskRoot):
    if not imgName.lower().endswith(".png"):
        continue

    input_path = os.path.join(maskRoot, imgName)
    output_path = os.path.join(distRoot, imgName)
    print(input_path)  # 打印路径，检查是否为：G:\data梯田\newlabels\001001001.png（无转义问题）
    # 读取并获取单通道
    im_data = read_png(input_path)
    single_band = im_data[0]  # 任意一个通道，值都相同

    # 计算距离变换（需为 binary）
    _, binary = cv2.threshold(single_band, 1, 255, cv2.THRESH_BINARY)
    dist_map = cv2.distanceTransform(binary, distanceType=cv2.DIST_L2, maskSize=3)

    # 归一化到 0-255 并转换为 uint8
    dist_map_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX)
    dist_map_uint8 = dist_map_norm.astype(np.uint8)

    # 复制三通道
    dist_map_3ch = np.stack([dist_map_uint8] * 3, axis=0)  # (3, H, W)

    # 写入结果
    write_png(output_path, dist_map_3ch)

print("所有PNG距离图生成完成。")
