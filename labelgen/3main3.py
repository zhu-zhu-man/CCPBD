# 3main3.py 影像裁剪与填充
import os
import argparse
import numpy as np
import cv2
from utils.io_utils import read_geotiff, write_geotiff_with_geo

def calculate_padding(h, w, crop_h, crop_w, overlap):
    step_h = int(crop_h * (1 - overlap))
    step_w = int(crop_w * (1 - overlap))
    pad_h = (step_h - (h % step_h)) % step_h
    pad_w = (step_w - (w % step_w)) % step_w
    return pad_h, pad_w

def adaptive_crop(image, mask, crop_h, crop_w, overlap, prefix, output_dir):
    h, w = image.shape[:2]
    pad_h, pad_w = calculate_padding(h, w, crop_h, crop_w, overlap)
    
    # 填充
    if len(image.shape) == 3:
        image_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    else:
        image_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    mask_pad = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    step_h = int(crop_h * (1 - overlap))
    step_w = int(crop_w * (1 - overlap))

    idx = 1
    for y in range(0, image_pad.shape[0] - crop_h + 1, step_h):
        for x in range(0, image_pad.shape[1] - crop_w + 1, step_w):
            img_crop = image_pad[y:y+crop_h, x:x+crop_w]
            mask_crop = mask_pad[y:y+crop_h, x:x+crop_w]

            img_name = f"{prefix}{idx:06d}.tif"
            mask_name = f"{prefix}{idx:06d}.png"

            cv2.imwrite(os.path.join(output_dir, 'images', img_name), img_crop)
            cv2.imwrite(os.path.join(output_dir, 'masks', mask_name), mask_crop)
            idx += 1

def main(args):
    tif_paths = np.load(os.path.join(args.config_path, 'tif_path.npy'))
    board_paths = np.load(os.path.join(args.config_path, 'board_path.npy'))
    output_mask_paths = np.load(os.path.join(args.config_path, 'output_path.npy'))

    for i in range(len(tif_paths)):
        print(f"Cropping image {i+1}/{len(tif_paths)}")
        # 读取裁剪后的影像（此处简化：假设已用boundary裁剪）
        img = read_geotiff(tif_paths[i])  # 实际应先用board裁剪
        mask = cv2.imread(output_mask_paths[i], cv2.IMREAD_GRAYSCALE)

        prefix = f"{i+1:03d}"
        adaptive_crop(
            img, mask,
            args.crop_size[0], args.crop_size[1],
            args.overlap,
            prefix,
            args.output_dir
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--crop_size', nargs=2, type=int, default=[512, 512])
    parser.add_argument('--overlap', type=float, default=0.2)
    args = parser.parse_args()
    main(args)