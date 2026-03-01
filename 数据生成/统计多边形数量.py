import cv2
import numpy as np
import glob
root_path = r'G:\data\shanxi'
root_path = r'G:\data\shanxi\newlabels'
img_path = glob.glob(root_path+'\\*.png')

number = 0
for i in range(len(img_path)):
    image = cv2.imread(img_path[i], cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image,127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_plgons = len(contours)
    number = number + num_plgons

print(number)