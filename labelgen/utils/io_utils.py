# utils/io_utils.py 工具函数
import logging
import numpy as np
from osgeo import gdal, ogr
import cv2

def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return logging.getLogger()

def rasterize_vector(vector_path, ref_image_path, burn_value=255):
    """ 使用参考影像栅格化矢量 """
    ref_ds = gdal.Open(ref_image_path)
    geo_transform = ref_ds.GetGeoTransform()
    x_size = ref_ds.RasterXSize
    y_size = ref_ds.RasterYSize
    projection = ref_ds.GetProjection()

    # 创建内存栅格
    mem_driver = gdal.GetDriverByName('MEM')
    target_ds = mem_driver.Create('', x_size, y_size, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    band = target_ds.GetRasterBand(1)
    band.Fill(0)
    band.SetNoDataValue(0)

    # 栅格化
    vector_ds = ogr.Open(vector_path)
    layer = vector_ds.GetLayer()
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[burn_value])

    array = target_ds.ReadAsArray()
    vector_ds = None
    target_ds = None
    ref_ds = None
    return array

def read_geotiff(path):
    ds = gdal.Open(path)
    channels = []
    for i in range(1, ds.RasterCount + 1):
        channels.append(ds.GetRasterBand(i).ReadAsArray())
    if len(channels) == 1:
        return channels[0]
    else:
        return np.stack(channels, axis=-1)

def write_geotiff_with_geo(path, array, ref_path):
    ref_ds = gdal.Open(ref_path)
    driver = gdal.GetDriverByName('GTiff')
    if len(array.shape) == 2:
        out_ds = driver.Create(path, array.shape[1], array.shape[0], 1, gdal.GDT_Byte)
        out_ds.GetRasterBand(1).WriteArray(array)
    else:
        out_ds = driver.Create(path, array.shape[1], array.shape[0], array.shape[2], gdal.GDT_Byte)
        for i in range(array.shape[2]):
            out_ds.GetRasterBand(i+1).WriteArray(array[:, :, i])
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    out_ds = None