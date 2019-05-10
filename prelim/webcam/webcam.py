"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Python cv2 interface.  Press <esc> to quit.
"""

import cv2


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,640);
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480);
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
