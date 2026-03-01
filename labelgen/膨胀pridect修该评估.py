#python3.7    UTF-8     PyCharm    time：2021.9.26.15.11

import numpy as np
from model12041 import *
import tensorflow as tf

from osgeo import gdal_array
import os
import skimage.io as io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# 自定义参数
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import init_ops
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils.generic_utils import to_list
import numpy as np
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.metrics.Kappa')
class Kappa(keras.metrics.Metric):
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name="Kappa",
                 dtype=None):
        super(Kappa, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        Acc = math_ops.div_no_nan(self.true_positives + self.true_negatives,
                                  self.true_negatives + self.true_positives + self.false_negatives + self.false_positives)
        Pe = math_ops.div_no_nan(
            (self.true_positives + self.false_positives) * (self.true_positives + self.false_negatives) + (
                        self.true_negatives + self.false_negatives) * (self.true_negatives + self.false_positives),
            (self.true_negatives + self.true_positives + self.false_negatives + self.false_positives) * (
                        self.true_negatives + self.true_positives + self.false_negatives + self.false_positives))
        # Kappa
        result = math_ops.div_no_nan(Acc - Pe,
                                     1 - Pe)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(Kappa, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    '''
       def __init__(self, name="Kappa", **kwargs):
        super(Kappa, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        # 在此处改代码
        # 该指标可以计算有多少样本被正确分类为属于给定类：
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        print('values',values)
        print('sample_weight',sample_weight)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")

            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0) 

    '''


@keras_export('keras.metrics.F1Score')
class F1Score(keras.metrics.Metric):
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name="F1Score",
                 dtype=None):
        super(F1Score, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        '''
         self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)         
        '''

        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                # metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        Recall = math_ops.div_no_nan(self.true_positives,
                                     self.true_positives + self.false_negatives)
        Precision = math_ops.div_no_nan(self.true_positives,
                                        self.true_positives + self.false_positives)

        # F1Score
        result = math_ops.div_no_nan(2 * Recall * Precision,
                                     Recall + Precision)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(F1Score, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    recall = tp/(tp + fn)
    precision = tp/(tp + fp)
    F1 = 2 * recall * precision/(recall + precision)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1Score: ', F1)

'''

需结合膨胀预测使用
第一次使用，循环中  shengcheng变量需要设置为1，用来生成对应预测影响  m参数用来控制阈值

'''

datapath = "D:\DeepLeaningDataspredict"
AA = np.load(datapath + '\\result_膨胀predict.npy')
BB = np.load(datapath + '\\result_膨胀label.npy')

truelabelpath = np.load("D:\DeepLeaningDatas" + "\\output_path.npy")

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
model.load_weights("G:\深度学习模型\Xumodel_2_my_model.h5")
METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      Kappa(name='Kappa'),
      F1Score(name='F1Score'),
      #tfa.metrics.CohenKappa(num_classes=2,name='kappa'),
      #tfa.metrics.F1Score(num_classes=2,name='F1'),
      #tf.keras.metrics.IoU(name='IoU'),
]
tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False,
    name='Adam'
)
model.compile(optimizer='Adam',loss= 'binary_crossentropy', metrics=METRICS)


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
                image = tf.io.decode_png(image,channels=3)

                #image11 = np.array(image11)
                image = tf.image.resize(image, (224, 224)) / 255.
                image = tf.expand_dims(image, 0)
                #image = tf.data.Dataset.from_tensor_slices(image)
                results = model.predict(image)

                image11 = image

                #test_labels = tf.io.read_file(BB[n])
                #test_labels = tf.io.decode_png(test_labels,channels=1)
                #test_labels = np.array(test_labels)
                #test_labels = tf.image.resize(test_labels, (224, 224)) / 255.
                #test_labels = tf.expand_dims(test_labels, 0)


                #results = tf.convert_to_tensor(results, tf.float32, name='results')

                # 创建会话


                '''
                test_labels = tf.make_tensor_proto(test_labels)
                test_labels = tf.make_ndarray(test_labels)
                results = np.reshape(results, [224, 224,1])
                test_labels  = np.reshape(test_labels ,[224,224,1])
                
                
                
                
                if ii ==1111 and iii == 39:   #寻找精度比较高的图像显示其混淆矩阵
                #if ii + 1 ==height[i] and iii + 1 ==width[i]:
                   # p = 0.5
                    #results = results.tolist()
                    #cm = confusion_matrix(test_labels, results)

                    weighted_results = model.evaluate(image11, test_labels, steps=1, verbose=0, use_multiprocessing=True)
                    for name, value in zip(model.metrics_names, weighted_results):
                        print(name, ': ', value)
                    print()
                





                    #plot_cm(test_labels, results)
                    cm = np.zeros((2,2))
                    cm[0][0] = weighted_results[3]
                    cm[0][1] = weighted_results[2]
                    cm[1][0] = weighted_results[4]
                    cm[1][1] = weighted_results[1]
                    #cm = confusion_matrix(labels, predictions > p)
                    plt.figure(figsize=(5, 5))
                    sns.heatmap(cm, fmt='.0f', annot=True)
                    #ddd =(iii+1)+(ii+1)*1000+(i+1)*1000000
                    plt.title('Confusion matrix @{:.2f}-@{:0>9d}'.format(0.5,(iii+1)+(ii+1)*1000+(i+1)*1000000))
                    #del ddd
                    plt.ylabel('Actual label')
                    plt.xlabel('Predicted label')
                    plt.show()
                    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
                    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
                    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
                    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
                    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
                    del cm
                '''

                #del weighted_results
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
            m = 0  #是否需要选取阈值，m=0，1，2
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
                    print('第%d张'%(n+1))
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
    truelabelarray =gdal_array.LoadFile(truelabelpath[i])

    out_band1 = truelabelarray[0, :, :]
    out_band2 = truelabelarray[1, :, :]
    out_band3 = truelabelarray[2, :, :]

    out_band1 = np.expand_dims(out_band1, axis=2)
    out_band2 = np.expand_dims(out_band2, axis=2)
    out_band3 = np.expand_dims(out_band3, axis=2)
    truelabel = np.concatenate((out_band1, out_band2, out_band3), axis=-1)

    plot_cm(truelabel, allpicture)

print('结束！！！！')




