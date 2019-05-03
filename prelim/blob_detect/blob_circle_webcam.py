import cv2
import numpy as np



def main():

    # Setup webcam
    cam = cv2.VideoCapture(0)

    # Setup SimpleBlobDetector to filter out circles of certain size
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500
    params.maxArea = 2000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.85

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)


    while True:

        # Read cam image and flip it to mirror
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)


        # Detect blobs.
        keypoints = detector.detect(img)


        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("Keypoints", im_with_keypoints)

        #cv2.imshow('Webcam', img)


        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
