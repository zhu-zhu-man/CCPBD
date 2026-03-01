# 删除带（1）的
# 将lable改成label
# 将(1)的删除
# 删除.shp.xml
# 删除.tif.aux.
# 删除.ovr
# 将boundry变成boundary
# 删除'CutTif'
# 删除'raster_clip'
# 删除Aexpend\Bexpend

import glob
import os

def get_all_files(directory):
    all_files = glob.glob(directory + '/**/*', recursive=True)
    return [file for file in all_files if os.path.isfile(file)]

directory = r"G:\BaiduNetdiskDownload\（分省）耕地地块标注（修改后副本）\北京市"  # 替换为你要搜索的文件夹路径
files = get_all_files(directory)
for file in files:
    if'.lock' in file  or '.shp.xml' in file or '.tif.aux.' in file or '.ovr' in file or 'CutTif' in file or 'raster_clip' in file or 'Aexpend' in file or 'Bexpend' in file:
        try:
            os.remove(file)
            print('删除',file)
        except:
            pass

    if '(1)' in file:
        name_1 = os.path.basename(file)
        if name_1.replace('(1)','') in str(files):
            try:
                os.remove(file)
                print('删除',file)
            except:
                pass
        else:
            try:
                os.rename(file,file.replace('(1)',''))
                print('重命名',file)
            except:
                pass
    if 'lable' in file:
        new_file_path = file.replace('lable','label')
        try:
            os.rename(file, new_file_path)
            print('重命名',file)
        except:
            pass

    if 'boundry.' in file:
        new_file_path = file.replace('boundry.','boundary.')
        try:
            os.rename(file, new_file_path)
            print('重命名',file)
        except:
            pass





