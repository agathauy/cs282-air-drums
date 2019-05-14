import matplotlib.pyplot as plt
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
bgs = cv2.createBackgroundSubtractorMOG2()

plt.figure()
plt.hold(True)
plt.axis([0,cap.get(3),cap.get(4),0])

count = 0   # for counting the number of frames
# Let us make an array for storing the values of (x,y) co-ordinates of the ball
# If the ball is not visible in the frame then keep that row as [-1.0,-1.0]
# Thus lets initialize the array with rows of [-1.0 , -1.0]
#numframes = cap.get(7)
numframes = 100
measuredTrack=np.zeros((int(numframes),2))-1
while count<(numframes):
    count+=1
    ret,img2 = cap.read()
    cv2.imshow("Video",img2)
    cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
    foremat=bgs.apply(img2,learningRate = 0.01)
    cv2.waitKey(20)
    ret,thresh = cv2.threshold(foremat,220,255,0)
    im2 , contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))  ## This prints the number of contours (foreground objects detected)
    if len(contours) > 0:
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])  ## Calculates the Area of contours
            if area > 100: ## We check this because the area of the ball is bigger than 100 and we want to plot that only
                m= np.mean(contours[i],axis=0) ### mean is taken for finding the centre of the contour (ball in this case)
                measuredTrack[count-1,:]=m[0]
                plt.plot(m[0,0],m[0,1],'xr')
    cv2.imshow('Foreground',foremat)
    cv2.namedWindow("Foreground",cv2.WINDOW_NORMAL)
    cv2.waitKey(80)
cap.release()
cv2.destroyAllWindows()
print(measuredTrack)
### save the trajectory of the ball in a numpy file , so that it can be used
### later to be passed as an input to the Kalman Filter process.
np.save("ballTrajectory", measuredTrack)
plt.axis((0,480,360,0))
plt.show()

