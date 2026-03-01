import os
import numpy as np
def clear():
    print('清除缓存开始')
    datapath = "D:\DeepLeaningDataspredict"
    datapath = "D:\DeepLeaningDatas"
    try:
        expend_path = np.load(datapath + '\\expend_path.npy')
        expend_path2 = np.load(datapath + '\\expend_path2.npy')
        for i in range(len(expend_path)):
            os.remove(expend_path[i])
            os.remove(expend_path2[i])
        del expend_path,expend_path2
        os.remove(datapath + '\\expend_path.npy')
        os.remove(datapath + '\\expend_path2.npy')
    except:
        a =0
    try:
        output_path = np.load(datapath + '\\output_path.npy')
        for i in range(len(output_path)):
            os.remove(output_path[i])
        del output_path
        os.remove(datapath + '\\output_path.npy')
    except:
        a =0
    try:
        resultAexpend_path = np.load(datapath + '\\resultAexpend_path.npy')
        resultBexpend_path = np.load(datapath + '\\resultBexpend_path.npy')
        for i in range(len(resultAexpend_path)):
            os.remove(resultAexpend_path[i])
            os.remove(resultBexpend_path[i])
        del resultAexpend_path,resultBexpend_path
        os.remove(datapath + '\\resultAexpend_path.npy')
        os.remove(datapath + '\\resultBexpend_path.npy')
    except:
        a =0
    try:
        os.remove(datapath + '\\tif_path.npy')
        os.remove(datapath + '\\house_path.npy')
    except:
        a =0
    print('清除完成')

clear()