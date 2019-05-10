import cv2
import numpy as np
import time


# Path of image to import
#path_image = '../data/blue_foil.png'
path_image = '../data/blue_640.png'
img = cv2.imread(path_image)


# Get 10x10 patch from center
patch_size = 10
patch_total_size = patch_size*patch_size

CALIBRATED = False

min_rgb = np.array([0, 0, 0])
max_rgb = np.array([0, 0, 0])

def cb_calibrate(event, x, y, flags, params):
    descriptor = []
    global CALIBRATED

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        line = "[x,y]: {},{} (rgb): ({}), (lab): ({})".format(point[0], point[1], img[y, x], img[y, x])
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

        print("Calibration Done!")
        CALIBRATED = True

if __name__ == '__main__':

    INIT = 0
    prev_pt = np.array([[0, 0]])
    new_pt = np.array([[0, 0]])

    cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,640);
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480);


    # Calculate FPS
    num_frames = 120
    start = time.time()
    for i in range(num_frames):
        ret, frame = cam.read()
    end = time.time()
    seconds = end - start
    FPS = num_frames / seconds
    print('FPS: %.2f'%FPS)

    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)

        # For calibration
        cv2.namedWindow('Calibrate Picture', cv2.WINDOW_NORMAL)
        #set mouse callback function for window
        cv2.setMouseCallback('Calibrate Picture', cb_calibrate)
        cv2.imshow('Calibrate Picture', img)


        if CALIBRATED == True:
            break
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
    print(min_rgb)
    print(max_rgb)

    # Start the blob detection
    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)

        # For calibration
        cv2.namedWindow('AirDrums', cv2.WINDOW_NORMAL)
        #set mouse callback function for window
        cv2.imshow('AirDrums', img)

        # Do the blob detection
        start = time.time()

        img = img

        maskLAB = cv2.inRange(img, min_rgb, max_rgb)
        #print(maskLAB)
        resultLAB = cv2.bitwise_and(img, img, mask = maskLAB)
        kernel = np.ones((10,10),np.uint8)
        dilation = cv2.dilate(maskLAB,kernel,iterations = 1)


        im2, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        height,width = dilation.shape[:2]
        img_contours = np.zeros((height,width,3), np.uint8)
        c_areas = []

        for c in contours:
            c_area = cv2.contourArea(c)
            c_areas.append(c_area)

        # Find max contour
        if (len(c_areas) != 0):
            max_c_area_index = c_areas.index(max(c_areas))
            print(max_c_area_index)
            # Calculate moment and center of contour
            M = cv2.moments(contours[max_c_area_index])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if INIT == 0:
                INIT = 1
                new_pt[0, 0] = cX
                new_pt[0, 1] = cY
                prev_pt[0, 0] = new_pt[0, 0]
                prev_pt[0, 1] = new_pt[0, 1]

            else:
                prev_pt[0, 0] = new_pt[0, 0]
                prev_pt[0, 1] = new_pt[0, 1]
                new_pt[0, 0] = cX
                new_pt[0, 1] = cY

            print(new_pt)
            print(prev_pt)
            dist = np.linalg.norm(new_pt-prev_pt)
            velocity = dist/(1/FPS)
            print('Velocity: %.2f pixels/second'%velocity)
            cv2.circle(img_contours, (cX, cY), 5, (255, 255, 255), -1)
        else:
            INIT = 0




        end = time.time()
        cv2.imshow("AirDrums: Centroid", img_contours)

        cv2.imshow("AirDrums: dilation", dilation)

        print("Seconds elapsed: {}".format(end-start))

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()










