import cv2
import numpy as np
import time




if __name__ == '__main__':

    INIT = 0
    prev_pt = np.array([[0, 0]])
    new_pt = np.array([[0, 10]])
    print(prev_pt.shape)
    cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,640);
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480);


    # Calculate FPS
    num_frames = 30
    start = time.time()
    for i in range(num_frames):
        ret, frame = cam.read()
    end = time.time()
    seconds = end - start
    FPS = num_frames / seconds
    print('FPS: %.2f'%FPS)

    print(new_pt)
    print(new_pt[0, 0])
    print(new_pt[0, 1])
    new_pt[0, 0] = 0
    new_pt[0, 1] = 20

    dist = np.linalg.norm(new_pt-prev_pt)
    print('Distance: {}'.format(dist))
    velocity = dist/(1/FPS)
    print('Velocity: %.2f pixels/second'%velocity)
