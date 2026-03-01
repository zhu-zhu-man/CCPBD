#python3.7    UTF-8     PyCharm    time：2021.9.26.15.11

import numpy as np
import time
import Crop_image膨胀空心label
import  return_and_creat_tif_path


datapath = "D:\DeepLeaningDatas"# 给到包含所有数据的文件夹，即：上层文件夹
cutw = 256#切割宽度
cuth = 256#切割高度
#return_and_creat_tif_path.returnpath(datapath)

for ii in range(3): #尝试打开文件
    try:
        #house_path = np.load(datapath + '\\house_path.npy')
        #tif_path = np.load(datapath + '\\tif_path.npy')
        output_path = np.load(datapath + '\\output_path.npy')[:21]
        #expend_path = np.load(datapath + '\\expend_path.npy')
        expend_path2 = np.load(datapath + '\\expend_path2.npy')[:21]
        print('打开成功，路径文件读取完成')
        break
    except:
        print('路径文件至少有一个未找到，重新生成二值图像，重新生成路径文件')
        #print('第%d尝试',(ii+1))
        #print('重新生成影像的路径文件')
        #return_and_creat_tif_path.returnpath(datapath)
        '这里需要一个函数（讲上面的代码做成一个函数，来调用'
        if ii == 2:
            print('已尝试三次，打开路径文件失败，请检查路径或文件')
        else:
            time.sleep(1)
overlap1 = 0.5  #膨胀率
overlap2 = 0.5  #膨胀率
AA = Crop_image膨胀空心label.cutting(cutw ,cuth ,overlap1 ,overlap2 ,output_path ,expend_path2)#切割遥感影像，小块路径返回
del output_path,expend_path2
np.save(datapath + '\\标签分割result_膨胀label.npy',AA)
