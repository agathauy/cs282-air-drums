import cv2
import numpy as np
import time
import sys
import logging
import pygame

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
        self.prev_velocities = [0, 0, 0]
        self.accelerations = [0, 0, 0]
        self.dir_vertical = [0, 0, 0]
        self.dir_horizontal = [0, 0, 0]
        # flag for detection
        self.flags = [0, 0, 0]

        # Init of camera
        self.cam = None

        # FPS
        self.FPS = 0
        # DELTA_T = 1/FPS
        self.DELTA_T = 0

        # Drum sounds
        self.ifDrumSoundsOn = False
        self.directory_sound = "../../sounds/"

        self.drum_snare = None
        self.drum_hihat = None
        self.drum_crash = None

        self.drum_tom1 = None
        self.drum_tom2 = None
        self.drum_ride = None

        self.drum_floor = None
        self.drum_bass = None



    def cb_calibrate(self, event, x, y, flags, params):
        '''
            Mouse callback for initial calibration of points
        '''
        img = params[0]
        item_num = params[1]
        descriptor = []

        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)
            line = "[x,y]: {},{} (rgb): ({})".format(point[0], point[1], img[y, x])
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

            # Get descriptor, going through rows
            for m in range(top_left[0], bottom_right[0] + 1):
                # Going through columns
                for n in range(top_left[1], bottom_right[1] + 1):
                    descriptor.append(img[m, n])
            descriptor = np.array(descriptor)
            logger.debug(descriptor)

            self.min_rgb[item_num, 0] = np.min(descriptor[:,0][self.patch_size:self.patch_total_size-self.patch_size])
            self.max_rgb[item_num, 0] = np.max(descriptor[:,0][self.patch_size:self.patch_total_size-self.patch_size])
            self.min_rgb[item_num, 1] = np.min(descriptor[:,1][self.patch_size:self.patch_total_size-self.patch_size])
            self.max_rgb[item_num, 1] = np.max(descriptor[:,1][self.patch_size:self.patch_total_size-self.patch_size])
            self.min_rgb[item_num, 2] = np.min(descriptor[:,2][self.patch_size:self.patch_total_size-self.patch_size])
            self.max_rgb[item_num, 2] = np.max(descriptor[:,2][self.patch_size:self.patch_total_size-self.patch_size])

            logger.debug("Calibration Done!")
            self.CALIBRATED = True

    def init_drum_sounds(self):
        # initialize pygame
        pygame.mixer.pre_init()
        pygame.init()
        self.drum_snare = pygame.mixer.Sound(self.directory_sound + "snare.wav")


    def init_calibrate(self):
        # Initialize calibrations
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
        self.DELTA_T = 1/self.FPS
        logger.debug('FPS: %.2f'%self.FPS)
        logger.debug('DELTA_T: %.2f'%self.DELTA_T)


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
                cv2.setMouseCallback(title, self.cb_calibrate, [img, i])
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
        # Check if no contours detected
        if (len(c_areas) != 0):
            max_c_area_index = c_areas.index(max(c_areas))
            logger.debug(max_c_area_index)
            # Calculate moment and center of contour
            M = cv2.moments(contours[max_c_area_index])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Check if there was detected pt in prev. frame
            if self.INIT_ITEM[item_num] == 0:
                # No detected contour in prev frame
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


            # Calculate dynamics given prev pt and new pt
            logger.debug(self.new_pt)
            logger.debug(self.prev_pt)
            dist = np.linalg.norm(self.new_pt-self.prev_pt)
            self.velocities[item_num] = dist/self.DELTA_T
            self.accelerations[item_num]= (self.velocities[item_num] - self.prev_velocities[item_num])/self.DELTA_T
            self.dir_vertical[item_num] = self.new_pt[item_num,1] - self.prev_pt[item_num,1]
            self.dir_horizontal[item_num] = self.new_pt[item_num,0] - self.prev_pt[item_num,0]

            logger.debug('Velocity: %.2f pixels/second'%self.velocities[item_num])
            logger.debug('Acceleration {}: {} pixels/second'.format(item, self.accelerations[item_num]))

            # Apply thresholding given acceleration
            # For left and right sticks
            if item != "Bass":
                if (self.accelerations[item_num] < -4000) and (self.dir_vertical[item_num] > 0) and (self.flags[item_num] < 0):
                    logger.debug('Acceleration {}: {} pixels/second'.format(item, self.accelerations[item_num]))
                    self.flags[item_num] = 20
                    if self.ifDrumSoundsOn == True:
                        # Detect area to determine which drum sound to play
                        # Temporary play snare
                        self.drum_snare.play()
                    self.prev_velocities[item_num] = self.velocities[item_num]
                    self.flags[item_num] = self.flags[item_num] - 1


            cv2.circle(img, (cX, cY), 5, self.blob_colors[item_num], -1)
            cv2.putText(img, item, (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



        else:
            self.INIT_ITEM[item_num] = 0
            #INIT_LEFT = 0




if __name__ == '__main__':
    drums = AirDrums()
    drums.init_drum_sounds()
    drums.init_calibrate()
    drums.playDrums()