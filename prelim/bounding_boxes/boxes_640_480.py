import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

# format: (x, y, width, height)
crash_loc = (int(0.00*width),int(0.00*height),int(0.34*width),int(0.34*height))
hihat_loc = (int(0.00*width),int(0.34*height),int(0.33*width),int(0.33*height))
snare_loc = (int(0.00*width),int(0.67*height),int(0.33*width),int(0.33*height))
tom1_loc = (int(0.34*width),int(0.34*height),int(0.33*width),int(0.33*height))
tom2_loc = (int(0.67*width),int(0.34*height),int(0.33*width),int(0.33*height))
ride_loc = (int(0.67*width),int(0.67*height),int(0.33*width),int(0.33*height))

while True:
    ret,im = cap.read()
    im = cv.flip(im,1)

    #cv.rectangle(im,(crash_loc[0],crash_loc[1]),(crash_loc[0]+crash_loc[2],crash_loc[1]+crash_loc[3]),(0,255,0),2)

    #cv.rectangle(im,(hihat_loc[0],hihat_loc[1]),(hihat_loc[0]+hihat_loc[2],hihat_loc[1]+hihat_loc[3]),(0,255,0),2)

    #cv.rectangle(im,(snare_loc[0],snare_loc[1]),(snare_loc[0]+snare_loc[2],snare_loc[1]+snare_loc[3]),(0,255,0),2)

    #cv.rectangle(im,(tom1_loc[0],tom1_loc[1]),(tom1_loc[0]+tom1_loc[2],tom1_loc[1]+tom1_loc[3]),(0,255,0),2)

    #cv.rectangle(im,(tom2_loc[0],tom2_loc[1]),(tom2_loc[0]+tom2_loc[2],tom2_loc[1]+tom2_loc[3]),(0,255,0),2)

    #cv.rectangle(im,(ride_loc[0],ride_loc[1]),(ride_loc[0]+ride_loc[2],ride_loc[1]+ride_loc[3]),(0,255,0),2)

    cv.line(im, (int(0.33*width), 0), (int(0.33*width), int(height)), (0,255,0), 1, 1)
    cv.line(im, (int(0.66*width), 0), (int(0.66*width), int(height)), (0,255,0), 1, 1)
    cv.line(im, (0, int(0.33*height)), (int(width), int(0.33*height)), (0,255,0), 1, 1)
    cv.line(im, (0, int(0.66*height)), (int(width), int(0.66*height)), (0,255,0), 1, 1)

    cv.imshow('img',im)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
