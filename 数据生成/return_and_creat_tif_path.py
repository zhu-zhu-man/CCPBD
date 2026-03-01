#python3.7    UTF-8     PyCharm    time：2021.9.26.15.11
import numpy as np
import pathlib
import raster_to_vector
import os, glob
def returnpath(datapath):  #读取、匹配、返回、生成路径。参数介绍：
    #datapath = "D:\DeepLeaningDataspredict"# 给到包含所有数据的文件夹，即：上层文件夹
    datapath = datapath
    # 矢量数据数据二值化-影像切割-生成词典部分
    #datapath = "D:\DeepLeaningDatas"  # 给到包含所有数据的文件夹，即：上层文件夹
    data_root = pathlib.Path(datapath) # datapath字符串对应的Windows path路径给data_path
    # all_paths1 = list(data_root.glob('*/**'))
    # all_paths1 = [str(Path) for Path in all_paths1]
    # 使用 glob 模块获取所有文件

    all_paths1 = glob.glob(os.path.join(data_root, '**', '*'), recursive=True)
    # 过滤掉文件夹，仅保留文件
    all_paths1 = [f for f in all_paths1 if os.path.isfile(f)]
    all_paths2 = []
    for si in range(len(all_paths1)):
        if '.ovr' not in all_paths1[si] and '.tif.xml' not in all_paths1[si] and '.tif.aux.' not in all_paths1[
            si] and 'expend' not in all_paths1[si] and 'CutTif' not in all_paths1[si] and 'raster_clip' not in \
                all_paths1[si] and '.tif' in all_paths1[si]:
            all_paths2.append(all_paths1[si])
        elif 'boundary.shp' in all_paths1[si] and '.shp.xml' not in all_paths1[si]:
            all_paths2.append(all_paths1[si])
        elif 'label.shp' in all_paths1[si] and '.shp.xml' not in all_paths1[si] or 'lable.shp' in all_paths1[
            si] and '.shp.xml' not in all_paths1[si]:
            all_paths2.append(all_paths1[si])
        else:
            if '.shp' in all_paths1[si] and '.shp.xml' not in all_paths1[si] or '.tif' in all_paths1[
                si] and '.ovr' not in all_paths1[si] and '.tif.xml' not in all_paths1[si] and '.tif.aux.' not in \
                    all_paths1[si] and 'CutTif' not in all_paths1[si] and 'raster_clip' not in all_paths1[
                si] and 'expend' not in all_paths1[si]:
                print(all_paths1[si])
                # 2025年7月30日 为东北数据裁剪添加新的约束条件
                if '样本' in all_paths1[si] and '.shp' in all_paths1[si] and '.shp.xml' not in all_paths1[si]:
                    all_paths2.append(all_paths1[si])
                elif '边界' in all_paths1[si] and '.shp' in all_paths1[si] and '.shp.xml' not in all_paths1[si]:
                    all_paths2.append(all_paths1[si])
                else:
                    print("请纠正以上数据文件名，然后再运行")
                    continue


    # all_paths2 = [str(item) for item in all_paths1 if ("Level" in item and "影像" not in item or "影像" in item and "Level" not in item or "耕地" in item or "边界" in item)]
    #由于上行命令获取的路径，无法同时带“影像”和“Level”，因此使用下面的循环，把有（有的没有）更深层次"影像\\Level19"的路径找出来并替换对应浅的路径
    i = len(all_paths2)
    # for ii in range(i):
    #     a = all_paths2[ii] + "\\Level"
    #     for iii in range(len(all_paths1)):
    #         if a in all_paths1[iii]:
    #             all_paths2[ii] = all_paths1[iii]

    #相似度检测,数据没问题后可以去掉
    def similar(str1, str2):
        str1 = str1 + ' ' * (len(str2) - len(str1))
        str2 = str2 + ' ' * (len(str1) - len(str2))
        return sum(1 if i == j else 0
                   for i, j in zip(str1, str2)) / float(len(str1))

    # for ii in range(0 , i, 3):
    #     b = similar(all_paths2[ii] ,all_paths2[ii+1])
    #     bb = similar(all_paths2[ii+2] ,all_paths2[ii+1])
    #     bbb = similar(all_paths2[ii] ,all_paths2[ii+2])
    #     b = (b+bb+bbb)/3
    #     if b < 0.75 :
    #         print(b)
    #         print(ii)
    #     print(similar('CharlesCC', 'Charles''\n'))
        #相似度检测，保证所有数据每组一致
    #每两个一组，但是每组获取到的路径顺序可能不同，有的影像在前，有的房屋在前，因此下面循环用来统一，保证影像在前，房屋在中，边界在后。
    # for ii in range(0 , i, 3):
    #     if "房屋" in all_paths2[ii]:
    #         exchange = all_paths2[ii]
    #         all_paths2[ii] = all_paths2[ii + 1]
    #         all_paths2[ii + 1] = exchange
    #     elif "房屋" in all_paths2[ii+2]:
    #         exchange = all_paths2[ii+2]
    #         all_paths2[ii+2] = all_paths2[ii + 1]
    #         all_paths2[ii + 1] = exchange
    #     if "界" in all_paths2[ii]:
    #         exchange = all_paths2[ii]
    #         all_paths2[ii] = all_paths2[ii + 2]
    #         all_paths2[ii + 2] = exchange
    ### 新添加判断
    for ii in range(0 , i, 3):
        if "样本" in all_paths2[ii+1] or "label" in all_paths2[ii+1] or "lable" in all_paths2[ii+1]:
        # if "边界" in all_paths2[ii+1]:
            exchange = all_paths2[ii+2]
            all_paths2[ii + 2] = all_paths2[ii+1]
            all_paths2[ii + 1] = exchange

    for ii in range(0 , i, 3):
        ast1 = False
        ast2 = False
        ast3 = False
        if ".tif" in all_paths2[ii]:
            ast1 = True
        if "y.shp" in all_paths2[ii+1] or "boundary" in all_paths2[ii+1] or "边界" in all_paths2[ii+1]:
        # if "y.shp" in all_paths2[ii+2] or "boundary" in all_paths2[ii+2] or "边界" in all_paths2[ii+2]:
            ast2 = True
        if "label" in all_paths2[ii+2] or "lable" in all_paths2[ii+2] or "样本" in all_paths2[ii+2]:
        # if "label" in all_paths2[ii+1] or "lable" in all_paths2[ii+1] or "样本" in all_paths2[ii+1]:
            ast3 = True
        if ast1 and ast2 and ast3:
            print('正确{}'.format(all_paths2[ii]))
        else:
            _a1=all_paths2[ii]
            _a2=all_paths2[ii+1]
            _a3=all_paths2[ii+2]
            print("错误{}".format(all_paths2[ii]))


    '''
    SAV = []
    for ii in range(0, i, 3):
        SAV.append( str(all_paths2[ii]+","+all_paths2[ii+1]+","+all_paths2[ii+2]))
    #SAV = np.array(all_paths2)
    np.savetxt("C:\\Users\\Administration\\Desktop\\核算\\学平.txt",SAV,fmt='%s',newline='\n')
    '''


    #顺序统一完成
    house_path = []
    tif_path = []
    board_path = []
    output_path = []
    expend_path =[] #用于指定和存储扩充后的tif影像
    expend_path2 =[]
    new_cut_tif_path = []
    # iaa = []
    # ibb = []
    # icc = []
    for ii in range(0 , i, 3):
        # aa = [str(path) for path in pathlib.Path(all_paths2[ii])]
        aa = all_paths2[ii]
        ia = 0
        iaa = []

        iaa.append(aa)

        # bb = [str(path) for path in pathlib.Path(all_paths2[ii + 1])]
        bb =all_paths2[ii+1]
        ib = 0
        ibb = []

        ibb.append(bb)
        # cc = [str(path) for path in pathlib.Path(all_paths2[ii + 2])]
        cc = all_paths2[ii+2]
        ic = 0
        icc = []

        icc.append(cc)


        if len(iaa) == 0 or len(ibb) == 0 or len(icc) == 0:
            print(all_paths2[ii])
            if len(iaa) == 0 and len(ibb) != 0 and len(icc) != 0:
                print("路径缺少影像数据")
            elif len(iaa) != 0 and len(iaa) == 0 and len(icc) != 0:
                print("路径缺少房屋数据")
            elif len(iaa) != 0 and len(ibb) != 0 and len(icc) == 0:
                print("路径缺少边界数据")
            elif len(iaa) == 0 and len(ibb) == 0 and len(icc) != 0:
                print("同时缺少影像和房屋数据")
            elif len(iaa) == 0 and len(iaa) != 0 and len(icc) == 0:
                print("路径缺少影像和边界数据")
            elif len(iaa) != 0 and len(ibb) == 0 and len(icc) == 0:
                print("路径缺少房屋和边界数据")

        elif len(iaa) == 1 and len(ibb) == 1 and len(icc) == 1:
            tif_path.append(iaa[0])
            house_path.append(ibb[0])
            board_path.append(icc[0])
            output_path.append(os.path.dirname(all_paths2[ii + 1]) + "\\raster_clip.tif")
            expend_path.append(os.path.dirname(all_paths2[ii + 1]) + "\\Aexpend.tif")
            expend_path2.append(os.path.dirname(all_paths2[ii + 1]) + "\\Bexpend.tif")
            new_cut_tif_path.append(os.path.dirname(all_paths2[ii + 1]) + "\\CutTif.tif")
        elif len(iaa) != len(ibb) or len(iaa) != len(icc) or len(ibb) != len(icc):
            print(all_paths2[ii])
            print('上路径图像和房屋数量不匹配')
        elif len(iaa) == len(ibb):
            for iiii in range(len(iaa)):
                tif_path.append(iaa[iiii])
                house_path.append(ibb[iiii])
                board_path.append(icc[iiii])
                output_path.append(os.path.dirname(all_paths2[ii + 1]) + "\\raster_clip"+str(iiii) + ".tif")
                expend_path.append(os.path.dirname(all_paths2[ii + 1]) + "\\Aexpend"+str(iiii) + '.tif')
                expend_path2.append(os.path.dirname(all_paths2[ii + 1]) + "\\Bexpend" + str(iiii) + '.tif')
                new_cut_tif_path.append(os.path.dirname(all_paths2[ii + 1]) + "\\CutTif" + str(iiii) + ".tif")
            print(all_paths2[ii])
            print('上路径出现多组合，已储存但未匹配，无法保证结果，注意检查')


    print('--------一致性检测开始---------')
    for ii in range(0 ,len(house_path) ,3):
        ss1 = ''
        #ss2 = ''
        str1 = house_path[ii].split("\\")
        for st1 in range(0 , len(str1)):
            if "耕地" in str1[st1]:
                for stt1 in range(0 ,st1 - 1):
                    if stt1 == 0:
                        sm1 = str1[stt1]
                    else:
                        sm1 = sm1 + str1[stt1]
                    if stt1 == st1 - 2:
                        ss1 = sm1
                break

        if ss1 in tif_path[ii]:
            print(house_path[ii])
            print(tif_path[ii])
            print("该路径匹配失败")
    print(("--------一致性检验完成---------"))

    for s in range(len(house_path)):
        # a = house_path[s]
        # b = output_path[s]
        # c = tif_path[s]#影像底，换了
        # d = board_path[s]
        # e = new_cut_tif_path[s]
        a = board_path[s]
        b = output_path[s]
        c = tif_path[s]#影像底，换了
        d = house_path[s]
        e = new_cut_tif_path[s]
        raster_to_vector.vector2raster( a, b, c, d, e,polygon2line = False)#看下方注释,20240901新增面砖呈线的控制参数,label 输入的依旧是面，线的厚度在函数内部设置
        # raster_to_vector.vector2raster( a, b, c, d, e,polygon2line = False)#看下方注释,20240901新增面砖呈线的控制参数,label 输入的依旧是面，线的厚度在函数内部设置
    print('二值图生成完成，切割后影像和对应路径生成完成')

    expend_path = np.array(expend_path)    #list转成ndarry
    expend_path2 = np.array(expend_path2) #list转成ndarry

    tif_path = new_cut_tif_path #将按边界切割后的影响路径给tif path

    '''
    现在将raster——to——vector。py文件中直接生成,按照board_path边界文件裁剪后的tif遥感影像。直接将tif_path原来的路径替换成新的。这样就不用增加新的文件
    raster_to_vector.vector2raster(a ,b ,c , d, e)
    e是临时路径，遍历完将其给c，这样原来写的就不用修改了
    '''

    np.save(datapath + '\\house_path.npy',house_path)#轮廓shp路径
    np.save(datapath + '\\output_path.npy',output_path)###########切边后二值影像
    np.save(datapath + '\\board_path.npy',board_path)#边界shp路径
    np.save(datapath + '\\tif_path.npy',tif_path)#影像tif路径
    np.save(datapath + '\\expend_path.npy',expend_path)#暂时没有实际影像，用于存放切割后的路径，IMG的切割后路径，现在将raster——to——vector。py文件中直接生成
    np.save(datapath + '\\expend_path2.npy',expend_path2)

    print('耕地+耕地二值+遥感影像文件对应路径已经生成,路径：%s'%(datapath))
