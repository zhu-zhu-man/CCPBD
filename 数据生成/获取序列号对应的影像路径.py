import numpy as np
import pandas as pd

datapath = r"G:\东三省农田标注\东三省农田标注副本\黑龙江省\平原"
out_path = r'G:\data'
save_file_path = datapath + '\\tif_path.npy'

# 加载路径数组
image_path = np.load(save_file_path)
print(image_path.shape)

# 将数组展平为一维（无论原始维度如何）
flat_paths = image_path.flatten()

# 创建序号列表（000格式）
indices = [f"{i:03d}" for i in range(len(flat_paths))]

# 创建DataFrame
df = pd.DataFrame({
    '序号': indices,
    '路径': flat_paths
})

# 保存到Excel文件
excel_path = out_path + '\\tif_paths.xlsx'
df.to_excel(excel_path, index=False)

print(f"Excel文件已保存至: {excel_path}")
print(f"共导出 {len(df)} 条记录")