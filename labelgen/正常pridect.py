#python3.7    UTF-8     PyCharm    time：2021.9.26.15.11

import numpy as np
from model8 import *
import tensorflow as tf
import os
import skimage.io as io

datapath = "D:\DeepLeaningDataspredict"
AA = np.load(datapath + '\\result_predict.npy')
hsize = 224
wsize = 224

numpicture = 0
width = []
height = []

for i in range(len(AA)):
    A = AA[i].split('\\')
    A = A[len(A)-1]#第几张+高+宽
    A = A.replace('.png','')#
    A = int(A)

    D = int(A/1000000)
    if D > numpicture:
        numpicture = D
        width.append(0)
        height.append(0)
    B = A % 1000
    if B > width[numpicture - 1]:
        width[numpicture - 1] = B
    C = int(A/1000)
    C = C % 1000
    if C > height[numpicture - 1]:
        height[numpicture - 1] = C
    del A,B,C

model = unet()
model.load_weights("D:\深度学习模型\\改进32modelWithWeight1.h5")

allpicture = []
for i in range(numpicture):
    allpicture = np.zeros((height[i] * hsize,width[i] * wsize))
    for ii in range(height[i]):
        for iii in range(width[i]):
            n = 0
            for iiii in range(numpicture):
                if iiii < numpicture - 1:
                    n += height[iiii] * width[iiii]
                else:
                    n += width[iiii] * ii +  iii
                    break
            image = tf.io.read_file(AA[n])
            image = tf.io.decode_png(image)
            image = tf.image.resize(image, (224, 224)) / 255.
            image = tf.expand_dims(image, 0)
            #image = tf.data.Dataset.from_tensor_slices(image)
            results = model.predict(image)
            results = np.reshape(results,[224,224])
            #
            cuttingresult = (AA[n].replace('result1', 'resultpridect'))
            img = results
            m = 0
            if m:
                img[img >= 0.25] = 1  # 此时1是浮点数，下面的0也是
                img[img < 0.25] = 0
            img = (img) * 255.0  # 将图像数据扩展到[0,255]
            img = np.array(img, dtype='uint8')
            io.imsave(os.path.join(cuttingresult), img)
            #上面为生成单张影像
            for jj in range(hsize):
                for jjj in range(wsize):
                    allpicture[ii * hsize + jj][iii * wsize + jjj] = img[jj][jjj]
            test = 0
    allpicture = np.array(allpicture, dtype='uint8')
    io.imsave(('D:\\DeepLeaningDataspredict\\'+ '%d.png'%(i+1)),allpicture)

print('结束！！！！')
'''
print(allpicture)




image_ds1 = []
for i in range(100):
    image = tf.io.read_file(AA[i])
    image = tf.io.decode_png(image)
    image = tf.image.resize(image,(224,224)) / 255.
    image_ds1.append(image)

image_ds1 = tf.expand_dims(image_ds1,1)
image_ds1 = tf.data.Dataset.from_tensor_slices(image_ds1)

# 导入模型
# model = NestedUNet(nClasses =1)
model = unet()
#image_ds1 = tf.expand_dims(image_ds1,0)
# 导入训练好的模型
#model.load_weights(r"D:\深度学习模型\\model1.h5")
#model.load_weights(r"D:\深度学习模型\120modelWithWeight1.h5")
model.load_weights("D:\深度学习模型\\改进7modelWithWeight1.h5")
results = model.predict(image_ds1)  # keras

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        cuttingresult = (AA[i].replace('result1', 'resultpridect'))
        #item = item
        #item = np.reshape(item,[224,224,])
        img=item[:,:,]
        #print(np.max(img),np.min(img))
        print(image)
        print(np.max(img), np.min(img))

        img[img>=0.25]=1#此时1是浮点数，下面的0也是
        img[img< 0.25]=0
        
        img = (img) * 255.0  # 将图像数据扩展到[0,255]
        img= np.array(img, dtype='uint8')
        print(np.max(img),np.min(img))
        #m = i +1
        #io.imsave(os.path.join(save_path,"%d_predict.png"%m),img)
        io.imsave(os.path.join(cuttingresult),img)

saveResult(datapath, results)  # data
print("over")
'''




