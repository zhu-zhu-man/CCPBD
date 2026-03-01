# 2main2.py 标签处理（防粘连 + 多模态标签）
import os
import argparse
import numpy as np
from osgeo import gdal, ogr, osr
from utils.io_utils import rasterize_vector, save_geotiff
from creat_edge_label import generate_edge_label
from creat_gaussian_distance_map import generate_gaussian_distance_map

def face_minus_edge_rasterize(label_shp, boundary_shp, output_mask, fill_value=255):
    """ 面减线：生成防粘连掩膜 """
    # 1. 面栅格化
    face_raster = rasterize_vector(label_shp, boundary_shp, burn_value=fill_value)
    
    # 2. 提取边界线
    line_shp = label_shp.replace('.shp', '_lines.shp')
    extract_polyline(label_shp, line_shp)
    
    # 3. 线栅格化
    edge_raster = rasterize_vector(line_shp, boundary_shp, burn_value=fill_value)
    
    # 4. 面减线
    final_mask = np.where((face_raster == fill_value) & (edge_raster != fill_value), fill_value, 0)
    return final_mask.astype(np.uint8)

def extract_polyline(polygon_shp, line_shp):
    """ 将面矢量转为线矢量 """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    src_ds = ogr.Open(polygon_shp)
    src_layer = src_ds.GetLayer()
    
    dst_ds = driver.CreateDataSource(line_shp)
    srs = src_layer.GetSpatialRef()
    dst_layer = dst_ds.CreateLayer('lines', srs, geom_type=ogr.wkbLineString)
    
    for feat in src_layer:
        geom = feat.GetGeometryRef()
        boundary = geom.GetBoundary()
        if boundary:
            out_feat = ogr.Feature(dst_layer.GetLayerDefn())
            out_feat.SetGeometry(boundary)
            dst_layer.CreateFeature(out_feat)
    
    dst_ds = None
    src_ds = None

def main(args):
    tif_paths = np.load(os.path.join(args.config_path, 'tif_path.npy'))
    board_paths = np.load(os.path.join(args.config_path, 'board_path.npy'))
    label_paths = np.load(os.path.join(args.config_path, 'label_path.npy'))
    output_paths = np.load(os.path.join(args.config_path, 'output_path.npy'))

    for i in range(len(tif_paths)):
        print(f"Processing label {i+1}/{len(tif_paths)}")
        mask = face_minus_edge_rasterize(
            label_paths[i], board_paths[i], output_paths[i]
        )
        # 保存标准掩膜
        cv2.imwrite(output_paths[i], mask)

        # 生成边缘标签
        if args.generate_edge:
            edge = generate_edge_label(mask, width=args.edge_width)
            edge_path = output_paths[i].replace('temp_mask', '../edges').replace('.png', '.png')
            cv2.imwrite(edge_path, edge)

        # 生成高斯距离图
        if args.generate_distance_map:
            dist_map = generate_gaussian_distance_map(mask, sigma=args.sigma)
            dist_path = output_paths[i].replace('temp_mask', '../dist_maps').replace('.png', '.png')
            cv2.imwrite(dist_path, dist_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--generate_edge', action='store_true', default=True)
    parser.add_argument('--edge_width', type=int, default=3)
    parser.add_argument('--generate_distance_map', action='store_true', default=True)
    parser.add_argument('--sigma', type=float, default=20.0)
    args = parser.parse_args()
    main(args)