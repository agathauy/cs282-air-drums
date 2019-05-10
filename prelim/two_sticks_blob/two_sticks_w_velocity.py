import cv2
import numpy as np
import time


# Path of image to import
#path_image = '../data/blue_foil.png'



# Get 10x10 patch from center
patch_size = 10
patch_total_size = patch_size*patch_size

CALIBRATED = False
# Left, right
CALIBRATIONS = [0, 0]


# left, right
min_rgb = np.array([[0, 0, 0], [0, 0, 0]])
max_rgb = np.array([[0, 0, 0], [0, 0, 0]])



def cb_calibrate(event, x, y, flags, params):
    descriptor = []
    global CALIBRATED
    global CALIBRATIONS
    global min_rgb, max_rgb

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
        if (CALIBRATIONS[0] == 0):
            min_rgb[0, 0] = np.min(list_l[patch_size:patch_total_size-patch_size])
            max_rgb[0, 0] = np.max(list_l[patch_size:patch_total_size-patch_size])
        else:
            min_rgb[1, 0] = np.min(list_l[patch_size:patch_total_size-patch_size])
            max_rgb[1, 0] = np.max(list_l[patch_size:patch_total_size-patch_size])

        print(line)


        print(descriptor[:, 1])
        line = "min: {}, max: {}, mean: {}".format(np.min(descriptor[:, 1]), np.max(descriptor[:, 1]), np.mean(descriptor[:, 1]))
        print(line)
        list_a = descriptor[:,1]
        list_a = np.sort(list_a, axis=None)
        line = "min: {}, max: {}, mean: {}".format(np.min(list_a[patch_size:patch_total_size-patch_size]), np.max(list_a[patch_size:patch_total_size-patch_size]), np.mean(list_a[patch_size:patch_total_size-patch_size]))
        if (CALIBRATIONS[0] == 0):
            min_rgb[0, 1] = np.min(list_a[patch_size:patch_total_size-patch_size])
            max_rgb[0, 1] = np.max(list_a[patch_size:patch_total_size-patch_size])
        else:
            min_rgb[1,1] = np.min(list_a[patch_size:patch_total_size-patch_size])
            max_rgb[1,1] = np.max(list_a[patch_size:patch_total_size-patch_size])
            
        print(line)

        print(descriptor[:, 2])
        line = "min: {}, max: {}, mean: {}".format(np.min(descriptor[:, 2]), np.max(descriptor[:, 2]), np.mean(descriptor[:, 2]))
        print(line)
        list_b = descriptor[:,2]
        list_b = np.sort(list_b, axis=None)
        line = "min: {}, max: {}, mean: {}".format(np.min(list_b[patch_size:patch_total_size-patch_size]), np.max(list_b[patch_size:patch_total_size-patch_size]), np.mean(list_b[patch_size:patch_total_size-patch_size]))
        if (CALIBRATIONS[0] == 0):
            min_rgb[0, 2] = np.min(list_b[patch_size:patch_total_size-patch_size])
            max_rgb[0, 2] = np.max(list_b[patch_size:patch_total_size-patch_size])
        else:
            min_rgb[1, 2] = np.min(list_b[patch_size:patch_total_size-patch_size])
            max_rgb[1, 2] = np.max(list_b[patch_size:patch_total_size-patch_size])
        print(line)

        print("Calibration Done!")
        CALIBRATED = True

if __name__ == '__main__':

    INIT_LEFT = 0
    INIT_RIGHT = 0
    TRIGGERED_LEFT = 0
    TRIGGERED_RIGHT = 0

    prev_pt = np.array([[0, 0]])
    new_pt = np.array([[0, 0]])

    cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,640);
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480);


    # Calculate FPS
    num_frames = 10
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
        cv2.namedWindow('Calibrate Picture: Left Stick', cv2.WINDOW_NORMAL)
        #set mouse callback function for window
        cv2.setMouseCallback('Calibrate Picture: Left Stick', cb_calibrate)
        cv2.imshow('Calibrate Picture: Left Stick', img)


        if CALIBRATED == True:
            break
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()
    CALIBRATED = False
    CALIBRATIONS[0] = 1
    print(min_rgb)
    print(max_rgb)


    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)

        # For calibration
        cv2.namedWindow('Calibrate Picture: Right Stick', cv2.WINDOW_NORMAL)
        #set mouse callback function for window
        cv2.setMouseCallback('Calibrate Picture: Right Stick', cb_calibrate)
        cv2.imshow('Calibrate Picture: Right Stick', img)


        if CALIBRATED == True:
            break
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

    CALIBRATIONS[1] = 1

    blob_color_left = (int(min_rgb[0, 0]), int(min_rgb[0, 1]), int(min_rgb[0, 2]))
    blob_color_right = (int(min_rgb[1, 0]), int(min_rgb[1, 1]), int(min_rgb[1, 2]))
    print('BLOB_COLORS')
    print(blob_color_left)
    print(blob_color_right)

    # Start the blob detection
    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)

        # For calibration
        #cv2.namedWindow('AirDrums', cv2.WINDOW_NORMAL)
        #set mouse callback function for window
        #cv2.imshow('AirDrums', img)

        # Do the blob detection
        start = time.time()


        # Detect for Left Stick
        maskLAB = cv2.inRange(img, min_rgb[0], max_rgb[0])
        #print(maskLAB)
        #resultLAB = cv2.bitwise_and(img, img, mask = maskLAB)
        kernel = np.ones((10,10),np.uint8)
        dilation = cv2.dilate(maskLAB,kernel,iterations = 1)


        im2, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        height,width = dilation.shape[:2]
        img_contours = img
        #img_contours = np.zeros((height,width,3), np.uint8)
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
            if INIT_LEFT == 0:
                INIT_LEFT = 1
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
            velocity_left = dist/(1/FPS)
            print('Velocity Left: %.2f pixels/second'%velocity_left)
            cv2.circle(img_contours, (cX, cY), 5, blob_color_left, -1)
            cv2.putText(img_contours, "Left", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        else:
            INIT_LEFT = 0


        #cv2.imshow("AirDrums: Dilation Left", dilation)



        # Detect for Right Stick
        maskLAB = cv2.inRange(img, min_rgb[1], max_rgb[1])
        #print(maskLAB)
        #resultLAB = cv2.bitwise_and(img, img, mask = maskLAB)
        kernel = np.ones((10,10),np.uint8)
        dilation = cv2.dilate(maskLAB,kernel,iterations = 1)


        im2, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        height,width = dilation.shape[:2]
        #img_contours = np.zeros((height,width,3), np.uint8)
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
            if INIT_RIGHT == 0:
                INIT_RIGHT = 1
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
            velocity_right = dist/(1/FPS)
            print('Velocity Right: %.2f pixels/second'%velocity_right)
            cv2.circle(img_contours, (cX, cY), 5, blob_color_right, -1)
            cv2.putText(img_contours, "Right", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        else:
            INIT_RIGHT = 0

        #cv2.imshow("AirDrums: Dilation Right", dilation


        # Check if velocity is considered as downwards




        end = time.time()
        cv2.imshow("AirDrums: Centroid", img_contours)


        print("Seconds elapsed: {}".format(end-start))

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    
    
    cv2.destroyAllWindows()










