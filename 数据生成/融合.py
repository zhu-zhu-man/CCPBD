from osgeo import gdal
import  os


a = 1  #是否需要将tif转为png
if a == 1:
    #ds = gdal.Open("D:\DeepLeaningDataspredict\山东建筑大学\房屋样本\Aexpend.tif")
    #ds = gdal.Open("D:\DeepLeaningDataspredict\海南省海口市秀英区城区-1\房屋样本\Aexpend.tif")
    ds = gdal.Open("D:\DeepLeaningDataspredict\黑龙江省哈尔滨市平房区城区\房屋样本\Aexpend.tif")
    driver = gdal.GetDriverByName('PNG')
    file1 = "D:\DeepLeaningDataspredict\测试.png"
    dst_ds = driver.CreateCopy(file1, ds)
    dst_ds = None
    del ds
    os.remove("D:\DeepLeaningDataspredict\测试.png.aux.xml" )
import skimage.io as io
import cv2
import numpy as np

img1 = io.imread("D:\DeepLeaningDataspredict\测试.png")  # 原图路径


print(img1.shape)
#img2 = gdal.Open("D:\DeepLeaningDataspredict\膨胀预测16-4-0.03.png")  # 掩膜图片路径，保证两个图片大小相同
img2 = gdal.Open("D:\DeepLeaningDataspredict\膨胀预测1.png")
width = img2.RasterXSize  # 获取数据宽度
height = img2.RasterYSize
img2 = img2.ReadAsArray(0, 0, width, height)
img2 = np.expand_dims(img2, axis=2)
img2 = np.concatenate((img2, img2, img2), axis=-1)


#1233
print(img2.shape)
dst = cv2.addWeighted(img1, 1, img2, 0.7, 0)  # 1和0.7是透明程度
io.imsave("D:\DeepLeaningDataspredict\green_only_out1.png", dst)  # 保存融合图片






