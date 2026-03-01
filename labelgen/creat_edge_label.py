# creat_edge_label.py 边缘标签生成
import cv2
import numpy as np

def generate_edge_label(mask, width=3):
    """ 使用膨胀-差分生成边缘 """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*width+1, 2*width+1))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    edge = dilated - mask
    return np.clip(edge, 0, 255).astype(np.uint8)

def generate_eroded_mask(mask, width=3):
    """ 腐蚀操作（可选） """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*width+1, 2*width+1))
    eroded = cv2.erode(mask, kernel, iterations=1)
    return eroded