import glob
from PIL import Image
import numpy as np
import os
from scipy.ndimage import binary_dilation

def new_lab(boundary_file, label_file, out_file0,out_file):
    # 读取两张影像
    image1 = Image.open(label_file)
    image2 = Image.open(boundary_file)
    # 将影像转换为numpy数组
    image1_np = np.array(image1)
    image2_np = np.array(image2)

    ### 形态学膨胀
    # 将灰度值二值化（0 和 1）
    image_np = image2_np[:,:,0]
    binary_image = (image_np > 128).astype(np.uint8)  # 将灰度图转换为0和1，假设二值化阈值为128
    # 定义膨胀的结构元素（3x3的方形结构元素）
    # selem = np.ones((3, 3), dtype=np.uint8)
    # 定义膨胀的十字形结构元素
    selem = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=np.uint8)
    # 使用 scipy.ndimage 的 binary_dilation 进行膨胀操作
    dilated_image = binary_dilation(binary_image, structure=selem, iterations=1).astype(np.uint8)
    # 将膨胀结果从 0/1 转换为 0/255 的格式
    dilated_image = dilated_image * 255
    image2_np = np.stack((dilated_image, dilated_image, dilated_image), axis=-1)
    # 将结果转换回Image格式
    result_image0 = Image.fromarray(image2_np.astype('uint8'))

    # 保存结果
    result_image0.save(out_file0)

    ### 膨胀完成
    # 检查两张影像是否具有相同的尺寸和波段
    if image1_np.shape != image2_np.shape:
        raise ValueError("两张影像的尺寸或波段数不一致")

    # 对应波段相减
    result_np = image1_np - image2_np

    # 将结果限制在有效的范围内（0-255）
    result_np = np.clip(result_np, 0, 255)

    # 将结果转换回Image格式
    result_image = Image.fromarray(result_np.astype('uint8'))

    # 保存结果
    result_image.save(out_file)

    print("相减后的影像已保存为 {}".format(out_file))


# root_path =r"D:\xjwdeeplearningdata\data"
root_path =r"G:\data"
boundary_path = root_path + "\\boundary"
label_path = root_path + "\\labels"
if not os.path.exists(root_path+"\\newlabels"):
    os.mkdir(root_path+"\\newlabels")
if not os.path.exists(root_path+"\\newboundary"):
    os.mkdir(root_path+"\\newboundary")
boundary_file = glob.glob(boundary_path+"\\*.png")
label_file = glob.glob(label_path+"\\*.png")
boundary_file.sort(key=lambda x: x.split('/')[-1].split('.png')[0])
label_file.sort(key=lambda x: x.split('/')[-1].split('.png')[0])
for i in range(len(boundary_file)):
    if os.path.basename(boundary_file[i]) == os.path.basename(label_file[i]):
        out_file = root_path+"\\newlabels\\"+ os.path.basename(boundary_file[i])
        out_file0 = root_path+"\\newboundary\\"+ os.path.basename(label_file[i])
        new_lab(boundary_file[i], label_file[i], out_file0,out_file)
    else:
        raise ValueError("影像不匹配")