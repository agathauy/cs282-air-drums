import cv2
import numpy as np
import time
import sys
import logging


# Configure logger
#logging.basicConfig(filename="test.log", format='%(filename)s: %(message)s', filemode='w')
logPath = "."
fileName = "test"
logging.basicConfig(
    level=logging.DEBUG,
    #format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(logPath, fileName)),
        logging.StreamHandler()
    ])
# Create a logger object
logger = logging.getLogger()

#console = logging.StreamHandler()
#console.setLevel(logging.DEBUG)
#logging.getLogger('').addHandler(console)
# Setting threshold level
#logger.setLevel(logging.DEBUG)

# # Use the logging methods
# logger.debug("This is a debug message")  
# logger.info("For your info")  
# logger.warning("This is a warning message")  
# logger.error("This is an error message")  
# logger.critical("This is a critical message")  



class AirDrums(object):
    def __init__(self):
        # Patch sizes
        # Get 10x10 patch from center
        self.patch_size = 10
        self.patch_total_size = self.patch_size*self.patch_size

        # RGB Calibration
        self.CALIBRATED = False
        # Left, right, bass
        self.CALIBRATIONS = [0, 0]
        self.blob_colors = [0, 0]
        self.min_rgb = np.array([[0, 0, 0], [0, 0, 0]])
        self.max_rgb = np.array([[0, 0, 0], [0, 0, 0]])

        # Points detected
        self.prev_pt = np.array([[0, 0], [0, 0]])
        self.new_pt = np.array([[0, 0], [0, 0]])
        self.INIT_ITEM = [0,0]

        # Velocities and accelerations
        self.velocities = [0, 0, 0]
        self.accelerations = [0, 0, 0]

        # Init of camera
        self.cam = None

        # FPS
        self.FPS = 0



    def cb_calibrate(self, event, x, y, flags, params):
        '''
            Mouse callback for initial calibration of points
        '''
        img = params[0]
        descriptor = []


        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)
            line = "[x,y]: {},{} (rgb): ({}), (lab): ({})".format(point[0], point[1], img[y, x], img[y, x])
            logger.debug(line)

            step_size = int(self.patch_size/2)
            height, width = img.shape[:2]

            # Turn into rows, cols format
            top_left = (point[1] - step_size, point[0] - step_size) 
            bottom_right = (point[1] + step_size, point[0] + step_size)

            # Check for top_left for out of bounds
            if (top_left[0] >= height) or (top_left[0] < 0) or (top_left[1] >= width) or (top_left[1] < 0):
                logger.debug("out of bounds")
                return
            if (bottom_right[0] >= height) or (bottom_right[0] < 0) or (bottom_right[1] >= width) or (bottom_right[1] < 0):
                logger.debug("out of bounds")
                return

            # Get descriptor
            #descriptor = []
            # Going through rows
            for m in range(top_left[0], bottom_right[0] + 1):
                # Going through columns
                for n in range(top_left[1], bottom_right[1] + 1):
                    descriptor.append(img[m, n])

            descriptor = np.array(descriptor)
            logger.debug(descriptor)

            logger.debug(descriptor[:, 0])
            line = "min: {}, max: {}, mean: {}".format(np.min(descriptor[:, 0]), np.max(descriptor[:, 0]), np.mean(descriptor[:, 0]))
            logger.debug(line)
            list_l = descriptor[:,0]
            list_l = np.sort(list_l, axis=None)
            #logger.debug(list_l[0:5])
            line = "min: {}, max: {}, mean: {}".format(np.min(list_l[self.patch_size:self.patch_total_size-self.patch_size]), np.max(list_l[self.patch_size:self.patch_total_size-self.patch_size]), np.mean(list_l[self.patch_size:self.patch_total_size-self.patch_size]))
            if (self.CALIBRATIONS[0] == 0):
                self.min_rgb[0, 0] = np.min(list_l[self.patch_size:self.patch_total_size-self.patch_size])
                self.max_rgb[0, 0] = np.max(list_l[self.patch_size:self.patch_total_size-self.patch_size])
            else:
                self.min_rgb[1, 0] = np.min(list_l[self.patch_size:self.patch_total_size-self.patch_size])
                self.max_rgb[1, 0] = np.max(list_l[self.patch_size:self.patch_total_size-self.patch_size])

            logger.debug(line)


            logger.debug(descriptor[:, 1])
            line = "min: {}, max: {}, mean: {}".format(np.min(descriptor[:, 1]), np.max(descriptor[:, 1]), np.mean(descriptor[:, 1]))
            logger.debug(line)
            list_a = descriptor[:,1]
            list_a = np.sort(list_a, axis=None)
            line = "min: {}, max: {}, mean: {}".format(np.min(list_a[self.patch_size:self.patch_total_size-self.patch_size]), np.max(list_a[self.patch_size:self.patch_total_size-self.patch_size]), np.mean(list_a[self.patch_size:self.patch_total_size-self.patch_size]))
            if (self.CALIBRATIONS[0] == 0):
                self.min_rgb[0, 1] = np.min(list_a[self.patch_size:self.patch_total_size-self.patch_size])
                self.max_rgb[0, 1] = np.max(list_a[self.patch_size:self.patch_total_size-self.patch_size])
            else:
                self.min_rgb[1,1] = np.min(list_a[self.patch_size:self.patch_total_size-self.patch_size])
                self.max_rgb[1,1] = np.max(list_a[self.patch_size:self.patch_total_size-self.patch_size])
                
            logger.debug(line)

            logger.debug(descriptor[:, 2])
            line = "min: {}, max: {}, mean: {}".format(np.min(descriptor[:, 2]), np.max(descriptor[:, 2]), np.mean(descriptor[:, 2]))
            logger.debug(line)
            list_b = descriptor[:,2]
            list_b = np.sort(list_b, axis=None)
            line = "min: {}, max: {}, mean: {}".format(np.min(list_b[self.patch_size:self.patch_total_size-self.patch_size]), np.max(list_b[self.patch_size:self.patch_total_size-self.patch_size]), np.mean(list_b[self.patch_size:self.patch_total_size-self.patch_size]))
            if (self.CALIBRATIONS[0] == 0):
                self.min_rgb[0, 2] = np.min(list_b[self.patch_size:self.patch_total_size-self.patch_size])
                self.max_rgb[0, 2] = np.max(list_b[self.patch_size:self.patch_total_size-self.patch_size])
            else:
                self.min_rgb[1, 2] = np.min(list_b[self.patch_size:self.patch_total_size-self.patch_size])
                self.max_rgb[1, 2] = np.max(list_b[self.patch_size:self.patch_total_size-self.patch_size])
            logger.debug(line)

            logger.debug("Calibration Done!")
            self.CALIBRATED = True

    def init_calibrate(self):
        self.computeFPS()
        self.colorCalibrations()

    def computeFPS(self):
        '''
            Calculate initial FPS
        '''
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

        # Calculate FPS
        num_frames = 10
        start = time.time()
        for i in range(num_frames):
            ret, frame = self.cam.read()
        end = time.time()
        seconds = end - start
        self.FPS = num_frames / seconds
        logger.debug('FPS: %.2f'%self.FPS)

    def colorCalibrations(self):
        '''
            Computes the color calibrations for the left and right sticks
        '''

        for i, val in enumerate(self.CALIBRATIONS):
            if i == 0:
                item = "Left Stick"
            elif i == 1:
                item = "Right Stick"
            else:
                item = "Bass Knee"

            while True:
                ret_val, img = self.cam.read()
                img = cv2.flip(img, 1)

                # For calibration
                title = "Calibrate Picture: {}".format(item)
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                #set mouse callback function for window
                cv2.setMouseCallback(title, self.cb_calibrate, [img])
                cv2.imshow(title, img)


                if self.CALIBRATED == True:
                    break
                if cv2.waitKey(1) == 27: 
                    sys.exit()
                    #break  # esc to quit
            cv2.destroyAllWindows()
            self.CALIBRATED = False
            self.CALIBRATIONS[i] = 1
            logger.debug(self.min_rgb)
            logger.debug(self.max_rgb)

        self.blob_colors[0] = (int(self.min_rgb[0, 0]), int(self.min_rgb[0, 1]), int(self.min_rgb[0, 2]))
        self.blob_colors[1] = (int(self.min_rgb[1, 0]), int(self.min_rgb[1, 1]), int(self.min_rgb[1, 2]))
        logger.debug('BLOB_COLORS')
        logger.debug(self.blob_colors[0])
        logger.debug(self.blob_colors[1])


    def playDrums(self):
        # Start the blob detection
        while True:
            ret_val, img = self.cam.read()
            img = cv2.flip(img, 1)

            start = time.time()
            # Find centroids for all blobs to be detected
            for i, val in enumerate(self.CALIBRATIONS):
                self.centroidDetection(img, i)

            end = time.time()
            logger.debug("Seconds elapsed: {}".format(end-start))
            cv2.imshow("AirDrums: Centroid", img)


            if cv2.waitKey(1) == 27: 
                # Press esc to quit
                sys.exit()


    def centroidDetection(self, img, item_num):
        item = None
        if item_num == 0:
            item = "Left"
        elif item_num == 1:
            item = "Right"
        elif item_num == 2:
            item = "Bass"
        else:
            logging.error("Invalid item_num")
            sys.exit()

        # Detect for blob
        maskLAB = cv2.inRange(img, self.min_rgb[item_num], self.max_rgb[item_num])
        kernel = np.ones((10,10),np.uint8)
        dilation = cv2.dilate(maskLAB,kernel,iterations = 1)


        im2, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        height,width = dilation.shape[:2]
        #img = np.zeros((height,width,3), np.uint8)
        c_areas = []

        for c in contours:
            c_area = cv2.contourArea(c)
            c_areas.append(c_area)

        # Find max contour
        if (len(c_areas) != 0):
            max_c_area_index = c_areas.index(max(c_areas))
            logger.debug(max_c_area_index)
            # Calculate moment and center of contour
            M = cv2.moments(contours[max_c_area_index])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if self.INIT_ITEM[item_num] == 0:
                self.INIT_ITEM[item_num] = 1
                self.new_pt[item_num, 0] = cX
                self.new_pt[item_num, 1] = cY
                self.prev_pt[item_num, 0] = self.new_pt[0, 0]
                self.prev_pt[item_num, 1] = self.new_pt[0, 1]

            else:
                self.prev_pt[item_num, 0] = self.new_pt[0, 0]
                self.prev_pt[item_num, 1] = self.new_pt[0, 1]
                self.new_pt[item_num, 0] = cX
                self.new_pt[item_num, 1] = cY

            logger.debug(self.new_pt)
            logger.debug(self.prev_pt)
            dist = np.linalg.norm(self.new_pt-self.prev_pt)
            self.velocities[item_num] = dist/(1/self.FPS)
            logger.debug('Velocity Left: %.2f pixels/second'%self.velocities[item_num])
            cv2.circle(img, (cX, cY), 5, self.blob_colors[item_num], -1)
            cv2.putText(img, item, (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        else:
            self.INIT_ITEM[item_num] = 0
            #INIT_LEFT = 0




if __name__ == '__main__':
    drums = AirDrums()
    drums.init_calibrate()
    drums.playDrums()