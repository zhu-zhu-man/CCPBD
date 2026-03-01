import os
import pandas as pd
from osgeo import ogr, gdal

def get_extent_shp(shp_path):
    ds = ogr.Open(shp_path)
    if not ds:
        return None
    layer = ds.GetLayer()
    extent = layer.GetExtent()
    spatial_ref = layer.GetSpatialRef()
    return extent, spatial_ref.ExportToWkt()

def get_extent_tif(tif_path):
    ds = gdal.Open(tif_path)
    if not ds:
        return None
    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    extent = (gt[0], gt[0] + cols * gt[1], gt[3] + rows * gt[5], gt[3])
    spatial_ref = ds.GetProjection()
    return extent, spatial_ref

def check_overlap(extent1, extent2):
    # extent1: (minx, maxx, miny, maxy)
    return not (extent1[1] < extent2[0] or extent1[0] > extent2[1] or
                extent1[3] < extent2[2] or extent1[2] > extent2[3])

def main(data_path):
    pairs = []
    for root, _, files in os.walk(data_path):
        shp_files = [f for f in files if f.lower().endswith(".shp")]
        tif_files = [f for f in files if f.lower().endswith(".tif")]
        for shp in shp_files:
            for tif in tif_files:
                pairs.append((os.path.join(root, shp), os.path.join(root, tif)))

    print(f"运行完发现共 {len(pairs)} 对 SHP + TIF 文件，开始检测...")

    results = []
    for shp_path, tif_path in pairs:
        shp_extent, shp_proj = get_extent_shp(shp_path)
        tif_extent, tif_proj = get_extent_tif(tif_path)

        coord_match = (shp_proj == tif_proj)
        overlap = check_overlap(
            (shp_extent[0], shp_extent[1], shp_extent[2], shp_extent[3]),
            (tif_extent[0], tif_extent[1], tif_extent[2], tif_extent[3])
        ) if shp_extent and tif_extent else False

        all_ok = coord_match and overlap
        results.append({
            "SHP路径": shp_path,
            "TIF路径": tif_path,
            "坐标系一致": coord_match,
            "范围重叠": overlap,
            "检测结果": "✅" if all_ok else "❌"
        })

    df = pd.DataFrame(results)
    output_path = os.path.join(data_path, "检查结果.xlsx")
    df.to_excel(output_path, index=False)
    print(f"✅ 检查完成，结果已保存到：{output_path}")

if __name__ == "__main__":
    datapath = r"G:\BaiduNetdiskDownload\（分省）耕地地块标注（修改后副本）\北京市"
    main(datapath)
