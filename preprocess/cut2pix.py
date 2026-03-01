# -*- coding: utf-8 -*-
"""
影像裁剪与地理位置获取工具
功能：裁剪GeoTIFF影像文件，生成边界矢量文件，并通过高德地图API获取影像中心点的省市县信息
"""

# 导入GDAL库用于处理地理空间栅格数据
from osgeo import gdal, osr, ogr
import glob
import pathlib
import numpy as np
import requests
from requests import exceptions

def getcountycoordinate(longitude, latitude):
    """
    GPS坐标转换为高德地图坐标系
    
    参数:
        longitude (float): GPS经度
        latitude (float): GPS纬度
    
    返回:
        str/int: 转换后的高德坐标（格式：经度,纬度），失败返回0
    """
    """
    GPS坐标转换为高德地图坐标系
    
    参数:
        longitude (float): GPS经度
        latitude (float): GPS纬度
    
    返回:
        str/int: 转换后的高德坐标（格式：经度,纬度），失败返回0
    """
    key = '963e729244dda24670e1717d1d826048'  # 高德地图API密钥
    url = 'https://restapi.amap.com/v3/assistant/coordinate/convert?'
    # 构建坐标转换请求URL
    url = url + '&locations=' + str(longitude) + ',' + str(latitude) +'&coordsys=gps' + '&key=' + key
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    try:
        # 发送HTTP请求，超时设置为连接3秒，读取7秒
        response = requests.get(url, headers = headers, timeout=(3,7))
    except exceptions.Timeout as e:
        print(str(e))
        return 0
    else:
        # 解析响应内容
        spl = str(response.content)
        spl = spl.split('"')
        if spl[3] =='1':  # 检查API返回状态码是否为1（成功）
            spl = spl[-2]
            return spl
        else:
            return 0


def getcounty(longitude_and_latitude):
    """
    根据经纬度坐标获取省市县信息（逆地理编码）
    
    参数:
        longitude_and_latitude (str): 经纬度坐标，格式："经度,纬度"
    
    返回:
        str/int: 省市县完整地址字符串，失败返回0
    """
    key = '963e729244dda24670e1717d1d826048'  # 高德地图API密钥
    url = 'https://restapi.amap.com/v3/geocode/regeo?'
    # 构建逆地理编码请求URL
    url = url + '&location=' + str(longitude_and_latitude) + '&key=' + key + '&extensions=base'
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers = headers, timeout=(3,7))
    except exceptions.Timeout as e:
        print(str(e))
        return 0
    else:
        # 解析响应并提取省市县信息
        spl = response.content.decode('utf8')
        spl = spl.split('"')
        if spl[3] == '1':  # 检查API返回状态码  # 检查API返回状态码
            # 定位省、市、区县字段在响应中的位置
            indexcity = spl.index('city')
            indexprovince = spl.index('province')
            insexdistrict = spl.index('district')
            # 根据不同情况组合省市县信息
            # 情况1：有省、市、区县
            if spl[indexcity + 2] != 'province' and spl[insexdistrict + 2] != 'towncode':
                county = spl[indexprovince + 2] + spl[indexcity + 2] + spl[insexdistrict + 2]
            # 情况2：直辖市（省=市），有区县
            elif spl[indexcity + 2] == 'province' and spl[insexdistrict + 2] != 'towncode':
                county = spl[indexprovince + 2] + spl[insexdistrict + 2]
            # 情况3：有省、市，无区县
            elif spl[indexcity + 2] != 'province' and spl[insexdistrict + 2] == 'towncode':
                county = spl[indexprovince + 2] + spl[indexcity + 2]
            # 情况4：只有省
            elif spl[indexcity + 2] == 'province' and spl[insexdistrict + 2] == 'towncode':
                county = spl[indexprovince + 2]
            return county



def readpath(datapath):
    """
    递归读取指定目录下所有TIF影像文件路径
    
    参数:
        datapath (str): 影像文件所在的根目录路径
    
    返回:
        list: 所有符合条件的TIF文件路径列表（排除包含"cut"的文件）
    """
    data_root = pathlib.Path(datapath)
    # 递归查找所有.tif文件
    all_paths1 = list(data_root.rglob('*.tif'))
    all_paths1 = [str(Path) for Path in all_paths1]
    # all_paths1 = [str(item) for item in all_paths1 if ("Level" in item)]
    # 过滤掉已经裁剪过的文件（路径中包含"cut"）
    all_paths1 = [str(item) for item in all_paths1 if ("cut" not in item)]
    return all_paths1

def creatdir_and_newtifpath(tifpath,size):
    """
    创建裁剪影像的输出目录结构并生成新影像路径
    
    参数:
        tifpath (list): 原始影像文件路径列表
        size (int): 裁剪尺寸（如1536、16384等）
    
    返回:
        tuple: (dirpath, newtifpath, npypath)
            - dirpath: 输出目录路径列表
            - newtifpath: 新影像文件路径列表
            - npypath: 保存省市县信息的文件路径
    """
    achapath1 = []  # 临时存储路径组成部分
    achapath2 = []  # 临时存储重组后的路径
    achapath3 = []  # 存储所有重组路径
    dirpath = []    # 输出目录列表
    newtifpath = [] # 新影像路径列表
    name = []       # 影像名称列表
    for i in range(len(tifpath)):
        # 分割路径字符串
        splitpath = tifpath[i].split('\\')
        # 获取除最后3级目录外的所有路径部分
        achapath1 = splitpath[0:len(splitpath)-3]
        name.append(splitpath[-3])  # 保存影像名称
        # 重新组装路径
        for ii in range(len(achapath1)):
            if ii == 0:
                achapath2 = achapath1[ii] + '\\'
            elif ii == len(achapath1) - 1:
                achapath2 = achapath2 + achapath1[ii]
            else:
                achapath2 = achapath2 + achapath1[ii]+ '\\'
        achapath3.append(achapath2)
        # 创建cutXXX主目录（如cut1536）
        if not pathlib.Path(achapath2 + '\\cut'+str(size)).exists():
            pathlib.Path.mkdir(pathlib.Path(achapath2 + '\\cut'+str(size)))
        # 创建影像名称子目录
        if not pathlib.Path(achapath2 + '\\cut'+str(size)+'\\' + splitpath[-3]).exists():
            pathlib.Path.mkdir(pathlib.Path(achapath2 + '\\cut'+str(size)+'\\' + splitpath[-3]))
        # 添加目录路径和新影像路径
        dirpath.append(achapath2 + '\\cut'+str(size)+'\\' + splitpath[-3])
        newtifpath.append(achapath2 + '\\cut'+str(size)+'\\' + splitpath[-3] + '\\' + splitpath[-1])
        # 第一次循环时创建省市县信息和名称文件路径
        if i == 0:
            npypath = achapath2 +'\\cut'+str(size)+'\\省市县.txt'
            namepath = achapath2 + '\\cut'+str(size)+'\\名称.txt'
        achapath2 = []  # 重置临时路径变量
    # 保存影像名称列表到文件
    np.savetxt(namepath, name, fmt='%s')

    return dirpath, newtifpath, npypath

def raster2poly(path, inband, outshp):
    """
    栅格数据转换为矢量多边形（Shapefile）
    
    参数:
        path (str): 源栅格文件路径（带坐标系的GeoTIFF）
        inband (numpy.ndarray): 输入的栅格矩阵数据
        outshp (str): 输出Shapefile文件路径
    
    功能:
        将栅格数据矢量化，生成多边形要素，主要用于生成影像边界范围
    """
    inraster = gdal.Open(path)  # 读取路径中的栅格数据
    projection = inraster.GetProjection()  # 获取投影信息
    geotrans = inraster.GetGeoTransform()  # 获取地理变换参数（仿射变换）
    # 创建内存中的临时栅格数据集
    memdrv = gdal.GetDriverByName('MEM')
    src_ds = memdrv.Create('', inband.shape[1], inband.shape[0], 1)
    src_ds.SetGeoTransform(geotrans)  # 设置地理变换参数
    src_ds.SetProjection(projection)   # 设置投影
    band = src_ds.GetRasterBand(1)
    band.WriteArray(inband)  # 写入矩阵数据
    inband = band
    # 创建空间参考对象
    prj = osr.SpatialReference()
    prj.ImportFromWkt(inraster.GetProjection())  # 从WKT格式导入投影信息

    # 创建Shapefile输出文件
    drv = ogr.GetDriverByName("ESRI Shapefile")
    Polygon = drv.CreateDataSource(outshp)  # 创建数据源
    Poly_layer = Polygon.CreateLayer(path[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)  # 创建图层（多边形类型）
    newField = ogr.FieldDefn('value', ogr.OFTReal)  # 创建属性字段，用于存储像素值（浮点型）
    Poly_layer.CreateField(newField)
    # 执行栅格矢量化（核心操作）
    gdal.Polygonize(inband, None, Poly_layer, 0, [])
    # gdal.FPolygonize(inband, None, Poly_layer, 0)  # 只转矩形，不合并
    Polygon.SyncToDisk()
    Polygon = None

    del inraster, projection, geotrans, memdrv, src_ds, band, prj, drv, Polygon, Poly_layer,

    # #删除value==0的要素（非房屋）
    #
    # gdal.AllRegister()
    # # 解决中文路径乱码问题
    # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # driver = ogr.GetDriverByName('ESRI Shapefile')
    # pFeatureDataset = driver.Open(outshp, 1)
    # pFeaturelayer = pFeatureDataset.GetLayer(0)
    #
    # # 按条件查询空间要素，本例查询字段名为Value，字段值为0的所有要素。
    # strValue = 0
    # strFilter = "Value = '" + str(strValue) + "'"
    # pFeaturelayer.SetAttributeFilter(strFilter)
    #
    # # 删除第二部查询到的矢量要素，注意，此时获取到的Feature皆为选择的Feature.
    # pFeatureDef = pFeaturelayer.GetLayerDefn()
    # pLayerName = pFeaturelayer.GetName()
    # pFieldName = "Value"
    # pFieldIndex = pFeatureDef.GetFieldIndex(pFieldName)
    # for pFeature in pFeaturelayer:
    #     pFeatureFID = pFeature.GetFID()
    #     pFeaturelayer.DeleteFeature(int(pFeatureFID))
    # strSQL = "REPACK " + str(pFeaturelayer.GetName())
    # pFeatureDataset.ExecuteSQL(strSQL, None, "")
    # pFeatureLayer = None
    # pFeatureDataset = None


def cuttif(tifpath, newtifpath, npypath, size):
    """
    裁剪TIF影像文件，生成边界矢量文件，并获取地理位置信息
    
    参数:
        tifpath (list): 原始影像文件路径列表
        newtifpath (list): 裁剪后影像保存路径列表
        npypath (str): 保存省市县信息的文件路径
        size (int): 裁剪尺寸（宽和高，单位：像素）
    
    功能:
        1. 从原始影像中心裁剪指定尺寸的区域
        2. 保留地理坐标信息
        3. 生成边界矢量文件
        4. 通过高德API获取影像中心点的省市县信息
    """
    npzcounty=[]  # 存储所有影像的省市县信息
    tifpath = np.array(tifpath)
    newtifpath = np.array(newtifpath)#转成ndarray,不转也可以
    boundary_shp_path = np.char.replace(np.array(newtifpath), '.tif', '_boundary.shp')#确定边界范围文件路径
    # zerotif = np.zeros((16384, 16384), dtype=np.uint8)
    zerotif = np.zeros((size, size), dtype=np.uint8)
    # 遍历所有影像文件
    for i in range(len(newtifpath)):
        isopen = 0
        try :
            in_ds = gdal.Open(tifpath[i])  # 尝试打开影像文件
            isopen = 1
        except:
            break  # 打开失败则跳出循环
        if isopen ==1:
            # 定义裁剪尺寸
            cs = size  # 宽度（列数）
            rs = size  # 高度（行数）
            width = in_ds.RasterXSize  # 获取数据宽度
            height = in_ds.RasterYSize  # 获取数据高度
            # if width < 16384 or height <16384:
            if width < size or height < size:
                print('影像宽或高小于' + str(size) +'，该影像不会进行裁剪！！！')
                break
            offset_x = int((width - cs)/2)
            offset_y = int((height-rs)/2)

            # 读取波段数据
            outbandsize = in_ds.RasterCount  # 获取波段数量
            datatype = in_ds.GetRasterBand(1).DataType  # 获取数据类型
            out_bands = []
            # 逐波段读取裁剪区域的数据
            for ii in range(outbandsize):
                out_bands.append(in_ds.GetRasterBand(ii + 1).ReadAsArray(offset_x, offset_y, cs, rs))

            out_bands = np.array(out_bands)

            # 创建GeoTIFF输出文件
            gtif_driver = gdal.GetDriverByName("GTiff")
            file = newtifpath[i]  # 输出文件路径
            # 创建裁剪后的影像文件（文件路径、宽、高、波段数、数据类型）
            out_ds = gtif_driver.Create(file, cs, rs, outbandsize, datatype)
            # 获取原影像的地理变换参数
            ori_transform = in_ds.GetGeoTransform()
            
            # 解析地理变换参数（仿射变换参数）
            top_left_x = ori_transform[0]  # 左上角X坐标（经度）
            w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
            top_left_y = ori_transform[3]  # 左上角Y坐标（纬度）
            n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率（通常为负值）
            
            # 根据裁剪偏移量计算新影像的左上角坐标
            top_left_x = top_left_x + offset_x * w_e_pixel_resolution
            top_left_y = top_left_y + offset_y * n_s_pixel_resolution
            
            # 构建新的地理变换参数元组
            dst_transform = (
            top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
            
            # 为裁剪后的影像设置地理坐标信息
            out_ds.SetGeoTransform(dst_transform)  # 设置地理变换
            out_ds.SetProjection(in_ds.GetProjection())  # 设置投影（与原影像相同）
            
            # 将裁剪后的波段数据写入输出文件
            for ii in range(outbandsize):
                out_ds.GetRasterBand(ii+1).WriteArray(out_bands[ii, :, :])
            out_ds.FlushCache()  # 将缓存数据写入磁盘
            
            # ========== 获取裁剪影像中心点的省市县信息 ==========
            # 计算裁剪后影像的中心点坐标
            offset_xx = cs / 2  # 中心点相对于左上角的X偏移
            offset_yy = rs / 2  # 中心点相对于左上角的Y偏移
            longitude = top_left_x + offset_xx * w_e_pixel_resolution  # 中心点经度
            latitude = top_left_y + offset_yy * n_s_pixel_resolution   # 中心点纬度
            
            # 调用高德API获取省市县信息
            longitude_and_latitude = getcountycoordinate(longitude, latitude)  # GPS坐标转高德坐标
            county = getcounty(longitude_and_latitude)  # 逆地理编码获取省市县
            npzcounty.append(county)  # 保存省市县信息

            # 释放内存
            del out_ds, out_bands, gtif_driver, cs, datatype, dst_transform, height, in_ds, n_s_pixel_resolution, rs
            del offset_y, offset_x, outbandsize, w_e_pixel_resolution, width, top_left_x, top_left_y
            
            # 输出处理进度
            print('第',i+1,'张影像裁剪完成')
            # 生成边界矢量文件
            raster2poly(file, zerotif, outshp=boundary_shp_path[i])
            print('第', i + 1, '张矢量文件已生成')
    
    # 将所有影像的省市县信息保存到文件
    np.savetxt(npypath, npzcounty, fmt='%s')

# ========== 主程序 ==========
if __name__ == "__main__":
    # 配置参数
    path = r"image_path"  # 当前使用的影像路径
    size = 1536  # 裁剪尺寸（像素）
    
    # 执行裁剪流程
    print("开始读取影像文件...")
    tifpath = readpath(path)  # 获取所有TIF文件路径
    print(f"共找到 {len(tifpath)} 个影像文件")
    
    print("创建输出目录结构...")
    dirpath, newtifpath, npypath = creatdir_and_newtifpath(tifpath, size)  # 创建目录和路径
    
    print("开始裁剪影像...")
    cuttif(tifpath, newtifpath, npypath, size)  # 执行裁剪
    print("\n所有影像处理完成！")