import cv2
import numpy as np
import time



def mouse_callback_calibrate(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        line = "[x,y]: {},{} (rgb): ({}), (lab): ({})".format(point[0], point[1], img[y, x], img_lab[y, x])
        print(line)

        step_size = int(patch_size/2)
        height, width = img.shape[:2]

        # Turn into rows, cols format
        top_left = (point[1] - step_size, point[0] - step_size) 
        bottom_right = (point[1] + step_size, point[0] + step_size)

        # Check for top_left for out of bounds
        if (top_left[0] >= height) or (top_left[0] < 0) or (top_left[1] >= width) or (top_left[1] < 0):
            print("out of bounds")
            return
        if (bottom_right[0] >= height) or (bottom_right[0] < 0) or (bottom_right[1] >= width) or (bottom_right[1] < 0):
            print("out of bounds")
            return

        # Get descriptor
        #descriptor = []
        # Going through rows
        for m in range(top_left[0], bottom_right[0] + 1):
            # Going through columns
            for n in range(top_left[1], bottom_right[1] + 1):
                descriptor.append(img[m, n])

        descriptor = np.array(descriptor)
        print(descriptor)

        print(descriptor[:, 0])
        line = "min: {}, max: {}, mean: {}".format(np.min(descriptor[:, 0]), np.max(descriptor[:, 0]), np.mean(descriptor[:, 0]))
        print(line)
        list_l = descriptor[:,0]
        list_l = np.sort(list_l, axis=None)
        #print(list_l[0:5])
        line = "min: {}, max: {}, mean: {}".format(np.min(list_l[patch_size:patch_total_size-patch_size]), np.max(list_l[patch_size:patch_total_size-patch_size]), np.mean(list_l[patch_size:patch_total_size-patch_size]))
        min_rgb[0] = np.min(list_l[patch_size:patch_total_size-patch_size])
        max_rgb[0] = np.max(list_l[patch_size:patch_total_size-patch_size])
        print(line)


        print(descriptor[:, 1])
        line = "min: {}, max: {}, mean: {}".format(np.min(descriptor[:, 1]), np.max(descriptor[:, 1]), np.mean(descriptor[:, 1]))
        print(line)
        list_a = descriptor[:,1]
        list_a = np.sort(list_a, axis=None)
        line = "min: {}, max: {}, mean: {}".format(np.min(list_a[patch_size:patch_total_size-patch_size]), np.max(list_a[patch_size:patch_total_size-patch_size]), np.mean(list_a[patch_size:patch_total_size-patch_size]))
        min_rgb[1] = np.min(list_a[patch_size:patch_total_size-patch_size])
        max_rgb[1] = np.max(list_a[patch_size:patch_total_size-patch_size])
        print(line)

        print(descriptor[:, 2])
        line = "min: {}, max: {}, mean: {}".format(np.min(descriptor[:, 2]), np.max(descriptor[:, 2]), np.mean(descriptor[:, 2]))
        print(line)
        list_b = descriptor[:,2]
        list_b = np.sort(list_b, axis=None)
        line = "min: {}, max: {}, mean: {}".format(np.min(list_b[patch_size:patch_total_size-patch_size]), np.max(list_b[patch_size:patch_total_size-patch_size]), np.mean(list_b[patch_size:patch_total_size-patch_size]))
        min_rgb[2] = np.min(list_b[patch_size:patch_total_size-patch_size])
        max_rgb[2] = np.max(list_b[patch_size:patch_total_size-patch_size])
        print(line)



class AirDrums(object):
    def __init__(self):

        # Patch sizes
        # Get 10x10 patch from center
        self.patch_size = 10
        self.patch_total_size = self.patch_size*self.patch_size
        self.descriptors = []

        # RGB Calibration
        self.min_rgb = np.array([0, 0, 0])
        self.max_rgb = np.array([0, 0, 0])


