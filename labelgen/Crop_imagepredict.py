# -*- coding: utf-8 -*-
from osgeo import gdal
import expendarray
from threading import Thread
import os

def cutting(crop_size1 ,crop_size2 ,Binary_imagepath ,cuttingpath):#参数介绍：（指定像素宽度,，指定像素高度,输入tif影像矩阵，输出扩充tif矩阵）
    cuttingresult_path = []
    for ii in range(len(Binary_imagepath)):
        in_ds = gdal.Open(Binary_imagepath[ii])
        # in_ds = gdal.Open(r'D:\graduate\GF1_WFV1_E120.0_N34.7_20180711_L1A0003315461.tiff')              # 读取要切的原图
        print("open tif file succeed")
        width = in_ds.RasterXSize                         # 获取数据宽度
        height = in_ds.RasterYSize                        # 获取数据高度
        outbandsize = in_ds.RasterCount                   # 获取数据波段数
        #im_geotrans = in_ds.GetGeoTransform()             # 获取仿射矩阵信息
        #im_proj = in_ds.GetProjection()                   # 获取投影信息
        datatype = in_ds.GetRasterBand(1).DataType
        #im_data = in_ds.ReadAsArray()                     #获取数据

        # 读取原图中的每个波段
        in_band1 = in_ds.GetRasterBand(1)
        in_band2 = in_ds.GetRasterBand(2)
        in_band3 = in_ds.GetRasterBand(3)
        #in_band4 = in_ds.GetRasterBand(4)

        # 定义切图的起始点坐标
        offset_x = 0
        offset_y = 0

        # 定义切图的大小（矩形框）
        size1 = crop_size1
        size2 = crop_size2
        col_num = int(width / size1)  #宽度可以分成几块
        row_num = int(height / size2) #高度可以分成几块
        if(width % size1 != 0):
            col_num += 1
        if(height % size2 != 0):
            row_num += 1


        #为了恰好切割区域，先对tif影像四个边界进行填充


        out_band1 = in_band1.ReadAsArray(offset_y, offset_x, width, height)
        out_band2 = in_band2.ReadAsArray(offset_y, offset_x, width, height)
        out_band3 = in_band3.ReadAsArray(offset_y, offset_x, width, height)


        #d多线程
        '''
        thread_list = []
        new1 = []
        new2 = []
        new3 = []
        '''
        cs = col_num * size2
        rs = row_num * size1

        class MyThread(Thread):
            def __init__(self,out_band ,cs ,rs):
                Thread.__init__(self)
                self.out_band = out_band
                self.cs = cs
                self.rs = rs
            def run(self):
                self.result = expendarray.expend(self.out_band ,self.cs ,self.rs)
            def get_result(self):
                return self.result

        t1 = MyThread(out_band1,cs,rs)
        t2 = MyThread(out_band2,cs,rs)
        t3 = MyThread(out_band3,cs,rs)
        #t1 = threading.Thread(target = expendarray.expend(out_band1,new1 ,col_num * size2,row_num * size1))
        #t2 = threading.Thread(target = expendarray.expend(out_band2,new2 ,col_num * size2, row_num * size1))
        #t3 = threading.Thread(target = expendarray.expend(out_band3,new3 ,col_num * size2, row_num * size1))
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()
        out_band1 = t1.get_result()
        out_band2 = t2.get_result()
        out_band3 = t3.get_result()

        '''
        for iiii in range(10000):
            if(new1 == [] or new2 == [] or new3 == []):
                time.sleep(0.5)
            else:
                break
        
        out_band1 = expendarray.expend(out_band1,col_num * size2,row_num * size1)#县宽后高
        out_band2 = expendarray.expend(out_band2,col_num * size2,row_num * size1)
        out_band3 = expendarray.expend(out_band3,col_num * size2,row_num * size1)
        '''


        gtif_driver = gdal.GetDriverByName("GTiff")
        file =cuttingpath[ii]  #需要指定路径
        # 创建切出来的要存的文件
        out_ds = gtif_driver.Create(file, col_num * size1, row_num * size2, outbandsize, datatype)  #（文件路径、宽、高、波段数、数据类型）
        # 获取原图的原点坐标信息
        ori_transform = in_ds.GetGeoTransform()
        '''
        if ori_transform:
            print(ori_transform)
            print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
            print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))
        '''
        # 读取原图仿射变换参数值
        top_left_x = ori_transform[0]  # 左上角x坐标
        w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
        top_left_y = ori_transform[3]  # 左上角y坐标
        n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率
        # 根据反射变换参数计算新图的原点坐标
        top_left_x = top_left_x - offset_x * w_e_pixel_resolution
        top_left_y = top_left_y - offset_y * n_s_pixel_resolution
        # 将计算后的值组装为一个元组，以方便设置
        dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
        # 设置裁剪出来图的原点坐标
        out_ds.SetGeoTransform(dst_transform)
        # 设置SRS属性（投影信息）
        out_ds.SetProjection(in_ds.GetProjection())


        out_ds.GetRasterBand(1).WriteArray(out_band1)
        out_ds.GetRasterBand(2).WriteArray(out_band2)
        out_ds.GetRasterBand(3).WriteArray(out_band3)

        out_ds.FlushCache()
        #print("FlushCache succeed")
        del out_ds, out_band1, out_band2, out_band3  # ,out_band4

        #这边就知道我们一共是分成了多少个 如果说有多余的 那我们就让那些也自己一小块好吧
        in_ds = gdal.Open(cuttingpath[ii])

        num = 1    #这个就用来记录一共有多少块的

        in_band1 = in_ds.GetRasterBand(1)
        in_band2 = in_ds.GetRasterBand(2)
        in_band3 = in_ds.GetRasterBand(3)
        width = in_ds.RasterXSize                         # 获取数据宽度
        height = in_ds.RasterYSize                        # 获取数据高度
        outbandsize = in_ds.RasterCount                   # 获取数据波段数
        im_geotrans = in_ds.GetGeoTransform()             # 获取仿射矩阵信息
        im_proj = in_ds.GetProjection()                   # 获取投影信息
        datatype = in_ds.GetRasterBand(1).DataType
        # 现在我们知道的是宽度是1304  高度是666
        #string1 = 'AJYF'
        #string2 = 'AJ'
        #if string2 in string1:
        #    string1.replace(string2, '')
        #if '.tif' in cuttingpath[ii]:
       # cuttingresult = [0]
        cuttingresult = (cuttingpath[ii].replace('.tif',''))

        # print("row_num:%d   col_num:%d" %(row_num,col_num))
        for i in range(row_num):    #从高度下手！！！ 可以分成几块！
            for j in range(col_num):
                offset_x = i * size1
                offset_y = j * size2
                ## 从每个波段中切需要的矩形框内的数据(注意读取的矩形框不能超过原图大小)
                b_ysize = min(width - offset_y, size2)
                b_xsize = min(height - offset_x, size1)

                #print("width:%d     height:%d    offset_x:%d    offset_y:%d     b_xsize:%d     b_ysize:%d" %(width,height,offset_x,offset_y, b_xsize, b_ysize))
                # print("\n")
                out_band1 = in_band1.ReadAsArray(offset_y, offset_x, b_ysize, b_xsize)
                out_band2 = in_band2.ReadAsArray(offset_y, offset_x, b_ysize, b_xsize)
                out_band3 = in_band3.ReadAsArray(offset_y, offset_x, b_ysize, b_xsize)
                # out_band4 = in_band4.ReadAsArray(offset_y, offset_x, b_ysize, b_xsize)
                # 获取Tif的驱动，为创建切出来的图文件做准备
                gtif_driver = gdal.GetDriverByName("GTiff")
                file = cuttingresult +'%04d.tif' %num
                #file = np.array(file) #list转成ndarry

                #file = r'C:\Users\Administrator\Desktop\last\%04d.tiff' % num
                num += 1
                # 创建切出来的要存的文件
                out_ds = gtif_driver.Create(file, b_ysize, b_xsize, outbandsize, datatype)
                #print("create new tif file succeed")

                # 获取原图的原点坐标信息
                ori_transform = in_ds.GetGeoTransform()
                '''
                if ori_transform:
                        print (ori_transform)
                        print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
                        print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))
                '''

                # 读取原图仿射变换参数值
                top_left_x = ori_transform[0]  # 左上角x坐标
                w_e_pixel_resolution = ori_transform[1] # 东西方向像素分辨率
                top_left_y = ori_transform[3] # 左上角y坐标
                n_s_pixel_resolution = ori_transform[5] # 南北方向像素分辨率

                # 根据反射变换参数计算新图的原点坐标
                top_left_x = top_left_x + offset_x * w_e_pixel_resolution
                top_left_y = top_left_y + offset_y * n_s_pixel_resolution

                # 将计算后的值组装为一个元组，以方便设置
                dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])

                # 设置裁剪出来图的原点坐标
                out_ds.SetGeoTransform(dst_transform)

                # 设置SRS属性（投影信息）
                out_ds.SetProjection(in_ds.GetProjection())

                # 写入目标文件
                out_ds.GetRasterBand(1).WriteArray(out_band1)
                out_ds.GetRasterBand(2).WriteArray(out_band2)
                out_ds.GetRasterBand(3).WriteArray(out_band3)
                # out_ds.GetRasterBand(4).WriteArray(out_band4)
                # 将缓存写入磁盘
                out_ds.FlushCache()
                #print("FlushCache succeed")
                del out_ds, out_band1, out_band2, out_band3

                #原来的
                file_path = file
                #return_file_path = "C:\data\images\\"
                return_file_path = "D:\DeepLeaningDataspredict\\result1\\"

                ds = gdal.Open(file_path)
                driver = gdal.GetDriverByName('PNG')
                file1 = return_file_path +'%09d.png' %(j+1+(i+1)*1000+(ii+1)*1000000)
                cuttingresult_path.append(file1)
                dst_ds = driver.CreateCopy(file1, ds)
                dst_ds = None
                src_ds = None
                del ds
                os.remove(file)
                os.remove(return_file_path +'%09d.png.aux.xml' %(j+1+(i+1)*1000+(ii+1)*1000000))
                del dst_ds,file,file1

                print('已累计',len(cuttingresult_path),'张','本图片第',num - 1,'张。')
    return cuttingresult_path