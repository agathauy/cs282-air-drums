import cv2
import numpy as np
import time


min_rgb = np.array([[0, 0, 0], [0, 0, 0]])
max_rgb = np.array([[0, 0, 0], [0, 0, 0]])


print(min_rgb)
print(min_rgb[0])

blob_color_left = (min_rgb[0, 0], min_rgb[0, 1], min_rgb[0, 2])
blob_color_right = (min_rgb[1, 0], min_rgb[1, 1], min_rgb[1, 2])

print(blob_color_left)
print(blob_color_right)