import cv2
import numpy as np
import time

# detect blue foil
# (ee298z) agatha:~/Documents/MSEE/CS282/cs282-air-drums/prelim/color_identify_mouse$ python color_identify.py
# [x,y]: 787,342 (rgb): ([129 105  80]), (lab): ([111 125 112])
# [x,y]: 787,342 (rgb): ([129 105  80]), (lab): ([111 125 112])
# [x,y]: 791,318 (rgb): ([174 136 112]), (lab): ([143 129 105])
# [x,y]: 761,334 (rgb): ([255 255 216]), (lab): ([248 116 124])
# [x,y]: 781,362 (rgb): ([122 100  77]), (lab): ([106 125 113])
# [x,y]: 820,354 (rgb): ([90 75 63]), (lab): ([ 80 127 118])
# [x,y]: 811,341 (rgb): ([171 144 115]), (lab): ([149 125 110])
# [x,y]: 800,318 (rgb): ([132  99  82]), (lab): ([107 130 108])
# [x,y]: 792,315 (rgb): ([157 123 101]), (lab): ([130 129 107])
# [x,y]: 779,351 (rgb): ([183 150 125]), (lab): ([156 127 108])
# [x,y]: 793,362 (rgb): ([122  94  76]), (lab): ([101 129 110])
# [x,y]: 801,341 (rgb): ([137 107  82]), (lab): ([113 127 109])
# [x,y]: 780,326 (rgb): ([218 179 153]), (lab): ([184 128 105])


#python 
#brightLAB = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
#darkLAB = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)

#img = cv2.imread("../data/blue_foil.png")
img = cv2.imread("../data/blue_640.png")

# blue_foil
# minLAB = np.array([133, 106, 81])
# maxLAB = np.array([167, 137, 110])

# blue 640
minLAB = np.array([109, 85, 63])
maxLAB = np.array([122, 97, 78])
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

maskLAB = cv2.inRange(img_lab, minLAB, maxLAB)
#print(maskLAB)
resultLAB = cv2.bitwise_and(img_lab, img_lab, mask = maskLAB)



kernel = np.ones((10,10),np.uint8)
#erosion = cv2.erode(maskLAB,kernel,iterations = 1)
dilation = cv2.dilate(maskLAB,kernel,iterations = 1)


#cv2.imshow("Output LAB", resultLAB)
#cv2.imshow("erosion", erosion)
end = time.time()
cv2.imshow("dilation", dilation)

print("Seconds elapsed: {}".format(end-start))
# Find contour 

# white = np.array([0, 0, 0])
# white_2 = np.array([10, 0, 0])

# mask = cv2.inRange(dilation, white, white_2)
# _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
#                                                     cv2.CHAIN_APPROX_NONE)
# blob = max(contours, key=lambda el: cv2.contourArea(el))
# M = cv2.moments(blob)
# center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
# contour = dilation.copy()
# cv2.circle(contour, center, 2, (0,0,255), -1)
# cv2.imshow("contour", contour)


cv2.waitKey(0)
cv2.destroyAllWindows()