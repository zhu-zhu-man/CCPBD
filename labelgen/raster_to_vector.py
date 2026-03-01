from osgeo import gdal,gdal_array
from osgeo import gdalconst
from osgeo import ogr


def polygon_to_line_in_memory(input_polygon_shapefile):# 20240901 新增，shp面转线
    print(input_polygon_shapefile)
    """
    将面状 Shapefile 转换为线状 Shapefile并存储在内存中
    """
    # 打开输入面 Shapefile
    shp = ogr.Open(input_polygon_shapefile)
    layer = shp.GetLayer()

    # 获取输入图层的空间参考系
    spatial_ref = layer.GetSpatialRef()

    # 创建内存数据源
    mem_driver = ogr.GetDriverByName('Memory')
    mem_datasource = mem_driver.CreateDataSource('')
    # 创建内存图层，并设置空间参考系
    mem_layer = mem_datasource.CreateLayer('memory_layer', geom_type=ogr.wkbLineString)
    # 复制输入图层的空间参考系到内存图层
    mem_layer.CreateField(ogr.FieldDefn('ID', ogr.OFTInteger))
    mem_layer.CreateField(ogr.FieldDefn('Name', ogr.OFTString))
    # 复制输入图层的字段定义到内存图层
    for field_defn in layer.schema:
        mem_layer.CreateField(field_defn)
    # 遍历面状要素，转换为线状要素并写入内存层
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is not None:
            line_geom = geom.Boundary()
            line_feature = ogr.Feature(mem_layer.GetLayerDefn())
            line_feature.SetGeometry(line_geom)
            # 复制属性字段
            for i in range(feature.GetFieldCount()):
                line_feature.SetField(feature.GetFieldDefnRef(i).GetNameRef(), feature.GetField(i))
            mem_layer.CreateFeature(line_feature)
            line_feature = None
    # 关闭输入文件和内存数据源
    shp = None
    # mem_datasource = None#不需要显式关闭，因为调用者会处理
    # 返回内存数据源和内存层
    return mem_datasource, mem_layer


def world2Pixel(geoMatrix, x, y):
  """
  Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
  the pixel location of a geospatial coordinate
  """
  ulX = geoMatrix[0]
  ulY = geoMatrix[3]
  xDist = geoMatrix[1]
  yDist = geoMatrix[5]
  rtnX = geoMatrix[2]
  rtnY = geoMatrix[4]
  pixel = int((x - ulX) / xDist)
  line = int((ulY - y) / xDist)
  return (pixel, line)

#
#  EDIT: this is basically an overloaded
#  version of the gdal_array.OpenArray passing in xoff, yoff explicitly
#  so we can pass these params off to CopyDatasetInfo
#


def vector2raster(inputfilePath,outputfile,templatefile,board_file,newtif_file,polygon2line = False):
    #inputfilePath = r"D:\DeepLeaningDatas\HaiNan\海南\白沙黎族自治县\海南省白沙黎族自治县邦溪镇\房屋样本\海南白沙黎族自治县邦溪镇房屋样本.shp"
    #outputfile = r'D:\DeepLeaningDatas\HaiNan\海南\白沙黎族自治县\海南省白沙黎族自治县邦溪镇\房屋样本\raster_clip.tif'
    #templatefile = r"D:\DeepLeaningDatas\HaiNan\海南\白沙黎族自治县\海南省白沙黎族自治县邦溪镇\影像\海南省白沙黎族自治县邦溪镇.tif"
    #下面这一部分是先生成裁剪后的遥感影像

    #shp_path = "D:\DeepLeaningDataspredict\黑龙江省哈尔滨市平房区城区\边界\黑龙江省哈尔滨市平房区城区.shp"
    #img_path = "D:\DeepLeaningDataspredict\黑龙江省哈尔滨市平房区城区\影像\黑龙江省哈尔滨市平房区城区.tif"
    #path = "D:\\DeepLeaningDataspredict\\黑龙江省哈尔滨市平房区城区\\范围切割后.tif"
    shapefile_path = board_file   #边界范围
    raster_path = templatefile    #遥感影像
    path = newtif_file
    srcArray = gdal_array.LoadFile(raster_path)
    srcImage = gdal.Open(raster_path)
    geoTrans = srcImage.GetGeoTransform()
    shapef = ogr.Open(shapefile_path)
    # lyr = shapef.GetLayer( os.path.split( os.path.splitext(shapefile_path)[0] )[1] )
    lyr = shapef.GetLayer()
    poly = lyr.GetNextFeature()
    minX, maxX, minY, maxY = lyr.GetExtent()
    ulX, ulY = world2Pixel(geoTrans, minX, maxY)
    lrX, lrY = world2Pixel(geoTrans, maxX, minY)
    # 下面为影像新的尺寸
    pxWidth = srcArray.shape[1]
    pxHeight = srcArray.shape[2]
    clip = srcArray[:, 0:pxWidth, 0:pxHeight]
    xoffset = ulX
    yoffset = ulY
    gtif_driver = gdal.GetDriverByName("GTiff")
    datatype = 1
    # out_ds = gtif_driver.Create(path, pxWidth, pxHeight, 3, datatype)
    out_ds = gtif_driver.Create(path, pxHeight, pxWidth, 3, datatype)


    # 获取原图的原点坐标信息
    ori_transform = srcImage.GetGeoTransform()
    '''
    if ori_transform:
            print (ori_transform)
            print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
            print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))
    '''

    # 读取原图仿射变换参数值
    top_left_x = ori_transform[0]  # 左上角x坐标
    w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
    top_left_y = ori_transform[3]  # 左上角y坐标
    n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率
    offset_x = 0
    offset_y = 0

    # 根据反射变换参数计算新图的原点坐标
    top_left_x = top_left_x + offset_x * w_e_pixel_resolution
    top_left_y = top_left_y + offset_y * n_s_pixel_resolution

    # 将计算后的值组装为一个元组，以方便设置
    dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])

    # 设置裁剪出来图的原点坐标
    out_ds.SetGeoTransform(dst_transform)

    # 设置SRS属性（投影信息）
    out_ds.SetProjection(srcImage.GetProjection())

    out_ds.GetRasterBand(1).WriteArray(clip[0, :, :])
    out_ds.GetRasterBand(2).WriteArray(clip[1, :, :])
    out_ds.GetRasterBand(3).WriteArray(clip[2, :, :])
    out_ds.FlushCache()
    del out_ds





    #下面这一部分是根据新生成的遥感影像，生成等尺寸对应的二值影像
    data = gdal.Open(path, gdalconst.GA_ReadOnly)
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    # vector = ogr.Open(inputfilePath)  #读取标注好的房屋shp文件
    # layer = vector.GetLayer()

    ##########
    # 根据polygon2line参数选择处理面还是线
    if polygon2line:
        vector, layer = polygon_to_line_in_memory(inputfilePath)  # 面转线，并获取内存中的数据
    else:
        vector = ogr.Open(inputfilePath)  # 直接使用输入的Shapefile
        layer = vector.GetLayer()

    ##########
    targetDataset = gdal.GetDriverByName('GTiff').Create(outputfile, x_res, y_res, 3, gdal.GDT_Byte)
    targetDataset.SetGeoTransform(data.GetGeoTransform())
    targetDataset.SetProjection(data.GetProjection())
    band = targetDataset.GetRasterBand(1)
    NoData_value = 0#-999
    band.SetNoDataValue(NoData_value)
    band.FlushCache()

    band1 = targetDataset.GetRasterBand(2)
    #NoData_value = 0#-999
    band1.SetNoDataValue(NoData_value)
    band1.FlushCache()
    band2 = targetDataset.GetRasterBand(3)
   # NoData_value = 0#-999
    band2.SetNoDataValue(NoData_value)
    band2.FlushCache()

    ###
    # 根据polygon2line参数选择处理面还是线
    if polygon2line: # 设定厚度
        line_thickness = 1
        options = ['ALL_TOUCHED=TRUE', f'LINE_THICKNESS={line_thickness}']
        gdal.RasterizeLayer(targetDataset, [1, 2, 3], layer, options=options)
    else:
        gdal.RasterizeLayer(targetDataset, [1, 2, 3], layer, )
    vector = None
    layer = None
