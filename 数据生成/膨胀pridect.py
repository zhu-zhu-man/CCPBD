#python3.7    UTF-8     PyCharm    time：2021.9.26.15.11

import numpy as np
from model12041 import *
import tensorflow as tf
import os
import skimage.io as io
'''

需结合膨胀预测使用
第一次使用，循环中  shengcheng变量需要设置为1，用来生成对应预测影响  m参数用来控制阈值

'''

datapath = "D:\DeepLeaningDataspredict"
AA = np.load(datapath + '\\result_膨胀predict.npy')
hsize = 224
wsize = 224
overlap1 = 0.5  #高的重叠率
overlap2 = 0.5
hstep = int(hsize * (1 - overlap1))   #一般取hize的一般，但是具体需要结合切割时的来选择，切割的重叠率两者应该需要保持一致
wstep = int(wsize * (1 - overlap2))

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

#model = creat_model()
model = Unet_Xception_ResNetBlock(1)
#model.load_weights("G:\深度学习模型\creatmodel_4_my_model.h5")
model.load_weights("G:\深度学习模型\\Xumodel_3-1_my_model.h5")
allpicture = []
for i in range(numpicture):
    #allpicture = np.zeros((height[i] * hsize,width[i] * wsize))
    allpicture = np.zeros((int(height[i] * hsize - hsize * (height[i] - 1) * overlap1), int(width[i] * wsize - wsize * (width[i] -1) * overlap2)))
    print(np.shape(allpicture))
    for ii in range(height[i]):
        for iii in range(width[i]):
            n = 0
            for iiii in range(numpicture):
                if iiii < numpicture - 1:
                    n += height[iiii] * width[iiii]
                else:
                    n += width[iiii] * ii +  iii
                    break
            '''
            
            '''
            shengcheng = 1   #shengcheng即生成，是否需要生成预测，需要值为1，不需要值为0

            l = 1  # 不需要预测值是0-255，需要预测值是0-1，所以引入l
            '''
            
            '''
            if shengcheng:

                image = tf.io.read_file(AA[n])
                image = tf.io.decode_png(image)
                image = tf.image.resize(image, (224, 224)) / 255.
                image = tf.expand_dims(image, 0)
                #image = tf.data.Dataset.from_tensor_slices(image)
                results = model.predict(image)
                #print(results)
                results = np.reshape(results,[224,224])
                del image
                img = results
            #

            cuttingresult = (AA[n].replace('result1', 'resultpridect'))
            if shengcheng == 0:
                img = io.imread(cuttingresult)
                l = 255.0
                #img = img[:,:,0]
            '''
            
            '''
            m = 1  #是否需要选取阈值，m=0，1，2
            '''
            
            '''
            if m == 1:
                img[img >= (0.5 *l )] = 1 * l  # 此时1是浮点数，下面的0也是
                img[img < (0.5*l )] = 0
            if m == 2:
                img[img !=0] = 1 * l
                img[img ==0] = 0



            if shengcheng == 1:
                img = (img) * 255.0  # 将图像数据扩展到[0,255]


            img = np.array(img, dtype='uint8')
            if shengcheng == 1:
                try:
                    io.imsave(os.path.join(cuttingresult), img)
                    print('第%d张'%n)
                except:
                    io.imsave(os.path.join(cuttingresult), img)
            #上面为生成单张影像
            k = int((hsize - hstep) * 0.5)
            kk = int((wsize - wstep) * 0.5)
            for jj in range(hstep):
                for jjj in range(wstep):
                    allpicture[int(ii * hstep + jj+ k)][int(iii * wstep + jjj + kk)] = img[jj + k][jjj + k]#ii表示第ii行，iii表示第iii列（不是影像像素的行列）
            del img
    allpicture = np.array(allpicture, dtype='uint8')
    io.imsave(('D:\\DeepLeaningDataspredict\\膨胀预测'+ '%d.png'%(i+1)),allpicture)

print('结束！！！！')




