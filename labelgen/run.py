#python3.7    UTF-8     PyCharm    time：2021.9.26.15.11
import os
import numpy as np
import time
import return_and_creat_tif_path
from threading import Thread
import clear

def load_and_preprocess_image(path, size=(224, 224)):  #加载并预处理图像函数
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, size) / 255.
    return image


'''
clear函数会根据已经生成的路径目录文件删除生成的图像文件，以及目录文件本身
如果需要调整切图像的大小，务必先使用此函数，删除之前的文件，避免冲突，留下碎片文件
'''


#clear.clear() #删除之前的文件


'''
为了以提升高效率，使用了多线程
多线程使用os.system提升效率
故：
    不容易传递参数
因此：
    如果需要调整切割图片的大小需要到2main2.py，和3main3.py中修改
    datapath需要更改，2main2.py和3main3.py中也需要修改

'''

datapath = "D:\DeepLeaningDatas"# 给到包含所有数据的文件夹，即：上层文件夹


'''


time1 = time.time()



return_and_creat_tif_path.returnpath(datapath)


class MyThread(Thread):
    def __init__(self, name):
        Thread.__init__(self)
        self.name = name

    def run(self):
        self.result = os.system(self.name)

t1 = MyThread('python 2main2.py')
t2 = MyThread('python 3main3.py')
t1.start()
t2.start()
t1.join()
t2.join()
time1 =time.time() -  time1
print('生成、切割影像和二值标签共耗时',time1,'秒')
del time1,t1,t2


'''

for ii in range(3): #尝试打开文件
    try:
        #house_path = np.load(datapath + '\\house_path.npy')
        #tif_path = np.load(datapath + '\\tif_path.npy')
        #output_path = np.load(datapath + '\\output_path.npy')
        #expend_path = np.load(datapath + '\\expend_path.npy')
        #expend_path2 = np.load(datapath + '\\expend_path2.npy')
        AA = np.load(datapath + '\\resultAexpend_path.npy')
        BB = np.load(datapath + '\\resultBexpend_path.npy')
        print('打开影像、标签路径文件成功')
        break
    except:
        print('影像、标签路径文件至少有一个未找到，重新生成二值图像，重新生成路径文件')
        print('第',ii+1,'次尝试')
        print('重新生成影像的路径文件')
        time1 = time.time()
        return_and_creat_tif_path.returnpath(datapath)
        t1 = MyThread('python 2main2.py')
        t2 = MyThread('python 3main3.py')
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        time1 = time.time() - time1
        print('生成、切割影像和二值标签共耗时', time1, '秒')
        del time1, t1, t2
        '这里需要一个函数（讲上面的代码做成一个函数，来调用'
        if ii == 2:
            print('已尝试三次，打开路径文件失败，请检查路径或文件')
        else:
            time.sleep(1)
print('匹配影像与标签影像路径')
print('影像分割小块数量有',len(AA),'块 * 2')



train_image = []
train_lable = []
test_image = []
test_lable = []
for ii in range(len(AA)):
    if ii < int(len(AA) * 0.7):
        train_image.append(AA[ii])
        train_lable.append(BB[ii])
    else:
        test_image.append(AA[ii])
        test_lable.append(BB[ii])






##########深度学习模型开始
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 2
EPOCHS = 30



##训练数据集
image_ds1 = tf.data.Dataset.from_tensor_slices(train_image) \
    .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# 上一行.map，表示的是image_ds.map，预处理函数
label_ds1 = tf.data.Dataset.from_tensor_slices(train_lable) \
    .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# 上一行.map，表示的是image_ds.map，预处理函数
image_label_ds1 = tf.data.Dataset.zip((image_ds1, label_ds1))
# 将图片和标签对应起来
image_count1 = len(train_image)
dataset1 = image_label_ds1.shuffle(buffer_size=image_count1) \
    .batch(batch_size=BATCH_SIZE) \
    .repeat(EPOCHS) \
    .prefetch(buffer_size=AUTOTUNE)


##测试数据集
image_ds2 = tf.data.Dataset.from_tensor_slices(test_image) \
    .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# 上一行.map，表示的是image_ds.map，预处理函数
label_ds2 = tf.data.Dataset.from_tensor_slices(test_lable) \
    .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# 上一行.map，表示的是image_ds.map，预处理函数
image_label_ds2 = tf.data.Dataset.zip((image_ds2, label_ds2))
# 将图片和标签对应起来
image_count2 = len(test_image)
dataset2 = image_label_ds2.shuffle(buffer_size=image_count2) \
    .batch(batch_size=BATCH_SIZE) \
    .repeat(EPOCHS) \
    .prefetch(buffer_size=AUTOTUNE)






ds_train, train_count = dataset1, image_count1


ds_test, test_count = dataset2, image_count2


'''
VGG16 = tf.keras.applications.VGG16(weights='imagenet',input_shape=(224,224,3),include_top=False)
VGG16.trainable = False
'''


'''

ResNet50 = tf.keras.applications.resnet_v2.ResNet50V2(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

ResNet50.trainable = False
model = tf.keras.models.Sequential([
    ResNet50,
    tf.keras.layers.GlobalAveragePooling2D(),
   # tf.keras.layers.Dense((224, 224,3) , activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
'''
epoch_steps = train_count // BATCH_SIZE
val_steps = test_count // BATCH_SIZE

'''


model.fit(ds_train, epochs=EPOCHS, steps_per_epoch=epoch_steps,validation_data=ds_test, validation_steps=val_steps, use_multiprocessing=True)
            #validation_data = ds_test, validation_steps = val_steps, use_multiprocessing = True)
# model ds_train 训练数据；epochs 迭代次数；step_per_epoch一个epoch包含的步数（每一步是一个batch的数据送入），当使用如TensorFlow数据Tensor之类的输入张量进行训练时，默认的None代表自动分割，即数据集样本数/batch样本数
# validation_data 验证数据集 validation_steps 验证步数
model.summary() #输出模型各层的参数状况


'''



from model import *
import matplotlib.pyplot as plt

# from myGenerator import load_train, load_test

model = unet()
# 这里调用load_train(csvDir, width, height, batch_size)产生数据
# 如果内存小batch_size就设为1吧
history = model.fit(ds_train,
                              steps_per_epoch=epoch_steps,
                              epochs=EPOCHS,
                              validation_data=ds_test ,
                              validation_steps=val_steps,
                              use_multiprocessing=True
                              )


model.save('modelWithWeight.h5')
model.save_weights('fine_tune_model_weight')

# print(history.history)

# 展示一下精确度的随训练的变化图
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 展示一下loss随训练的变化图
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()














print('Finish!!')







