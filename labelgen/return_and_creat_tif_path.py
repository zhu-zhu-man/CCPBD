# return_and_creat_tif_path.py 路径智能匹配
import os
import numpy as np

def is_boundary_file(name):
    return any(kw in name.lower() for kw in ['boundary', '边界', '样本'])

def is_label_file(name):
    return any(kw in name.lower() for kw in ['label', 'lable', '样本'])

def match_paths(root_dir, meta_dir):
    tif_list, board_list, label_list = [], [], []

    for root, _, files in os.walk(root_dir):
        tifs = [os.path.join(root, f) for f in files if f.endswith('.tif')]
        shps = [os.path.join(root, f) for f in files if f.endswith('.shp')]

        # 按文件名前缀分组（简化版：假设同目录或同前缀）
        for tif in tifs:
            base = os.path.splitext(os.path.basename(tif))[0]
            matched_board = None
            matched_label = None

            for shp in shps:
                shp_base = os.path.splitext(os.path.basename(shp))[0]
                if is_boundary_file(shp_base) and (base in shp_base or shp_base in base):
                    matched_board = shp
                if is_label_file(shp_base) and (base in shp_base or shp_base in base):
                    matched_label = shp

            if matched_board and matched_label:
                tif_list.append(tif)
                board_list.append(matched_board)
                label_list.append(matched_label)

    # 保存路径列表
    np.save(os.path.join(meta_dir, 'tif_path.npy'), tif_list)
    np.save(os.path.join(meta_dir, 'board_path.npy'), board_list)
    np.save(os.path.join(meta_dir, 'label_path.npy'), label_list)

    # 初始化输出路径占位符
    output_masks = [os.path.join(meta_dir, 'temp_mask', f"{i:09d}.png") for i in range(len(tif_list))]
    os.makedirs(os.path.join(meta_dir, 'temp_mask'), exist_ok=True)
    np.save(os.path.join(meta_dir, 'output_path.npy'), output_masks)