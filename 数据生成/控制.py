
import time
import return_and_creat_tif_path
from threading import Thread
import os
from PIL import Image

# datapath = "D:\DeepLeaningDatas"# 给到包含所有数据的文件夹，即：上层文件夹
# datapath = r"G:\BaiduNetdiskDownload\耕地地块标注20240831"
datapath = r"G:\BaiduNetdiskDownload\（分省）耕地地块标注（修改后副本）\江西省\坡耕地"
# datapath = r"G:\东三省农田标注\东三省农田标注副本"

time1 = time.time()
return_and_creat_tif_path.returnpath(datapath)  # 在函数内部设置是否将面转成线，以及线的厚度

class MyThread(Thread):
    def __init__(self, name):
        Thread.__init__(self)
        self.name = name

    def run(self):
        self.result = os.system(self.name)
'''t1 = MyThread('python 2main2.py')
t2 = MyThread('python 3main3.py')
t1.start()
t2.start()
t1.join()
t2.join()
time1 =time.time() -  time1
print('生成、切割影像和二值标签共耗时',time1,'秒')
del time1,t1,t2'''
# 第一遍，只能生成一种标签，是掩膜标签，需要在return_and_creat_tif_path.returnpath修改是面还是线
t1 = MyThread('python 2main2.py')
# t2 = MyThread('python 3main3.py')
t1.start()
# t2.start()
t1.join()
# t2.join()
time1 =time.time() -  time1
print('生成、切割影像和二值标签共耗时',time1,'秒')
del time1,t1

# 第二遍，只生成另一种标签，是边缘标签


# 处理完后还需要用py 对两个标签相减
