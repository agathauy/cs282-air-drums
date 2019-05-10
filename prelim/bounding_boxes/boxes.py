import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# precompute some points
x1 = int(0.33*width)
x2 = int(0.66*width)
y1 = int(0.33*height)
y2 = int(0.66*height)

while True:
    ret,im = cap.read()
    im = cv.flip(im,1)

    point = (x1,y1)

    cv.line(im, (x1, 0), (x1, height), (0,255,0), 1, 1)
    cv.line(im, (x2, 0), (x2, height), (0,255,0), 1, 1)
    cv.line(im, (0, y1), (width, y1), (0,255,0), 1, 1)
    cv.line(im, (0, y2), (width, y2), (0,255,0), 1, 1)

    (x,y) = point
    if x1 < x and x < x2 and y1 < y and y < y2:
        print('Hello World')

    cv.imshow('img',im)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
