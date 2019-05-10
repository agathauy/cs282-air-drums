import cv2
import numpy as np
import time

# Path of image to import
#path_image = '../data/blue_foil.png'
path_image = '../data/green_blob.png'
#img = NULL
#img_lab= NULL
img = cv2.imread(path_image)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


# Get 10x10 patch from center
patch_size = 10
patch_total_size = patch_size*patch_size


min_rgb = np.array([0, 0, 0])
max_rgb = np.array([0, 0, 0])
#descriptor = []
CALIBRATED = False

def mouse_callback(event, x, y, flags, params):
    global CALIBRATED
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
        descriptor = []
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

        print("Calibration Done!")

        CALIBRATED = True
        #cv2.imshow('Patch 100', descriptor)


if __name__ == '__main__':

    while True:
        cv2.namedWindow('Calibrate Picture', cv2.WINDOW_NORMAL)

        #set mouse callback function for window
        cv2.setMouseCallback('Calibrate Picture', mouse_callback)

        cv2.imshow('Calibrate Picture', img)
        if CALIBRATED == True:
            break
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()
    print(min_rgb)
    print(max_rgb)


    # Start the blob detection


    cv2.namedWindow('AirDrums', cv2.WINDOW_NORMAL)
    #set mouse callback function for window
    cv2.imshow('AirDrums', img)

    # Do the blob detection
    start = time.time()

    #img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_lab = img

    # minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
    # maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])

    # minLAB = np.array([100, 127, 105])
    # maxLAB = np.array([200, 130, 124])


    # raw
    # minLAB = np.array([125, 96, 72])
    # maxLAB = np.array([255, 244, 221])

    # With lowest 10 and top 10 samples cut off

    maskLAB = cv2.inRange(img_lab, min_rgb, max_rgb)
    #print(maskLAB)
    resultLAB = cv2.bitwise_and(img_lab, img_lab, mask = maskLAB)



    kernel = np.ones((10,10),np.uint8)
    #erosion = cv2.erode(maskLAB,kernel,iterations = 1)
    dilation = cv2.dilate(maskLAB,kernel,iterations = 1)


    #cv2.imshow("Output LAB", resultLAB)
    #cv2.imshow("erosion", erosion)
    end = time.time()
    cv2.imshow("AirDrums: dilation", dilation)

    print("Seconds elapsed: {}".format(end-start))


        # find contours in the binary image
    im2, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    height,width = dilation.shape[:2]
    img_contours = np.zeros((height,width,3), np.uint8)
    c_areas = []
    for c in contours:
        # calculate moments for each contour
        #M = cv2.moments(c)

        c_area = cv2.contourArea(c)
        c_areas.append(c_area)
        #print(c_area)

        # # calculate x,y coordinate of center
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])
        # cv2.circle(img_contours, (cX, cY), 5, (255, 255, 255), -1)
        # cv2.putText(img_contours, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Find max contour
    max_c_area_index = c_areas.index(max(c_areas))
    print(max_c_area_index)
    M = cv2.moments(contours[max_c_area_index])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(img_contours, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(img_contours, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Centroid", img_contours)
        #if cv2.waitKey(1) == 27: 
        #    break  # esc to quit
    cv2.waitKey(0)

    cv2.destroyAllWindows()





