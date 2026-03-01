# run.py 主控调度程序
import os
import argparse
import yaml
import numpy as np
from return_and_creat_tif_path import match_paths
from utils.io_utils import setup_logger

def main(config):
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'images'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'masks'), exist_ok=True)
    if config.get('generate_edge', True):
        os.makedirs(os.path.join(config['output_dir'], 'edges'), exist_ok=True)
    if config.get('generate_distance_map', True):
        os.makedirs(os.path.join(config['output_dir'], 'dist_maps'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'metadata'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'logs'), exist_ok=True)

    logger = setup_logger(os.path.join(config['output_dir'], 'logs', 'processing.log'))

    # 步骤1: 路径匹配
    logger.info("Step 1: Matching input paths...")
    match_paths(config['root_dir'], os.path.join(config['output_dir'], 'metadata'))

    # 步骤2: 标签处理
    logger.info("Step 2: Processing labels...")
    os.system(f"python 2main2.py --config_path {os.path.join(config['output_dir'], 'metadata')}")

    # 步骤3: 影像处理与裁剪
    logger.info("Step 3: Processing images and cropping...")
    os.system(f"python 3main3.py --config_path {os.path.join(config['output_dir'], 'metadata')} --crop_size {config['crop_size'][0]} {config['crop_size'][1]} --overlap {config['overlap_ratio']}")

    logger.info("All done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    main(config)