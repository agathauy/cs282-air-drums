

import cv2

#the [x, y] for each right-click event will be stored here
#right_clicks = list()

#this function will be called whenever the mouse is right-clicked

path_image = '../data/blue_foil.png'
img = cv2.imread(path_image)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def mouse_callback(event, x, y, flags, params):

    #right-click event value is 2
    if event == 1:
        global right_clicks

        #store the coordinates of the right-click event
        #right_clicks.append([x, y])

        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        #print(right_clicks)
        line = "[x,y]: {},{} (rgb): ({}), (lab): ({})".format(x, y, img[y, x], img_lab[y, x])
        print(line)





cv2.namedWindow('image', cv2.WINDOW_NORMAL)

#set mouse callback function for window
cv2.setMouseCallback('image', mouse_callback)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

