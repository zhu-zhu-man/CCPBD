#!/usr/bin/env python
# coding=utf-8
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_float
from skimage.morphology import binary_erosion, rectangle, disk, binary_dilation
import glob, os
import pathlib
import numpy as np


from osgeo import gdal_array




def create_edge_labels(seg_labels_path, edge_labels_path,erosion_num,dilation_num):  #输入路径、输出路径、二进制形态学腐蚀次数、二进制形态学膨胀次数
    for i in range(len(seg_labels_path)):
        im = gdal_array.LoadFile(seg_labels_path[i])

        # img1 = imread(seg_labels_path[i])
        # im = rgb2gray(imread(seg_labels_path[i]))




        threshold = 0.5
        im[im < threshold] = 0
        im[im >= threshold] = 1
        for _ in range(dilation_num):  #必须先膨胀后腐蚀
            im = binary_dilation(im)
        im1 = im
        if dilation_num > 0:
            for _ in range(erosion_num + dilation_num + 1):
                im1 = binary_erosion(im1)
        else:
            for _ in range(erosion_num + dilation_num):
                im1 =binary_erosion(im1)

        boundary = im.astype(np.uint8) - im1.astype(np.uint8)
        boundary = boundary.astype(np.uint8)
        boundary = np.expand_dims(boundary * 255,axis=-1)
        boundary = np.concatenate((boundary, boundary, boundary),axis=-1)
        imsave(edge_labels_path[i], boundary[0,:,:,:],  check_contrast=False)
        if i % 200 ==0:
            print('已生成' + str(i) + '张')


# def main(seg_labels_path,erosion_num,dilation_num):
#     seg_labels_path = seg_labels_path
#     all_seg_labels_path = glob.glob(seg_labels_path + '\\*.tif')
#     all_seg_labels_path.sort(key=lambda x: x.split('/')[-1].split('.tif')[0])  # 排序
#     all_edge_labels_path = []
#     for i in range(len(all_seg_labels_path)):
#         all_edge_labels_path.append(all_seg_labels_path[i].replace('label','edge'))
#     if not pathlib.Path(seg_labels_path.replace('label','edge')).exists():
#         os.mkdir(seg_labels_path.replace('label','edge'))
#     # if '.tif' in all_edge_labels_path[0]:
#     #     for i in range(len(all_edge_labels_path)):
#     #         all_edge_labels_path[i] = all_edge_labels_path[i].replace('.tif','.png')
#     create_edge_labels(all_seg_labels_path,all_edge_labels_path,erosion_num,dilation_num)
def main(seg_labels_path,erosion_num,dilation_num):
    seg_labels_path = seg_labels_path
    all_seg_labels_path = glob.glob(seg_labels_path + '\\*.png')
    all_seg_labels_path.sort(key=lambda x: x.split('/')[-1].split('.png')[0])  # 排序
    all_edge_labels_path = []
    for i in range(len(all_seg_labels_path)):
        all_edge_labels_path.append(all_seg_labels_path[i].replace('label','edge'))
    if not pathlib.Path(seg_labels_path.replace('label','edge')).exists():
        os.mkdir(seg_labels_path.replace('label','edge'))
    # if '.tif' in all_edge_labels_path[0]:
    #     for i in range(len(all_edge_labels_path)):
    #         all_edge_labels_path[i] = all_edge_labels_path[i].replace('.tif','.png')
    create_edge_labels(all_seg_labels_path,all_edge_labels_path,erosion_num,dilation_num)

if __name__ == '__main__':
    # seg_labels_path = "C:\\XJWDeepLearningData\\whudata256\\train\\labels"
    # seg_labels_path = "G:\农田提交裁剪\label"
    seg_labels_path = "D:\\xjwdeeplearningdata\\iFLYTEK\\preliminary256\\train\label"
    erosion_num = 3#腐蚀次数
    dilation_num = 0#膨胀次数
    main(seg_labels_path,erosion_num,dilation_num)