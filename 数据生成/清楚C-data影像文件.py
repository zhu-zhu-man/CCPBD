import numpy as np
import os


C = "C:\data\images_path.npy"
D = "C:\data\labels_path.npy"
datapath1 = "D:\DeepLeaningDatas"
A = datapath1 +'\\resultAexpend_path.npy'
B = datapath1 +'\\resultBexpend_path.npy'
try:
    expend_path = np.load(A)
    expend_path2 = np.load(B)
    for i in range(len(expend_path)):
        os.remove(expend_path[i])
        os.remove(expend_path2[i])
    del expend_path, expend_path2
    os.remove(A)
    os.remove(B)
except:
    print('清除失败')


try:
    os.remove(C)
    os.remove(D)
except:
    a = 0

