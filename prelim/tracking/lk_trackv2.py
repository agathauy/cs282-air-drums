import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)

# Calculate FPS
num_frames = 120
start = time.time()
for i in range(num_frames):
    ret, frame = cap.read()
end = time.time()
seconds = end - start
FPS = num_frames / seconds
print('FPS: %.2f'%FPS)

# Create old frame
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Lucas kanade params
lk_params = dict(winSize = (15,15),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)
point_selected = False
point = ()
old_points = np.array([[]])
while True:
    start = time.time()
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if point_selected is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        dt = 1/FPS
        dy = new_points[0,1] - old_points[0,1]
        dx = new_points[0,0] - old_points[0,0]
        vy = dy/dt
        vx = dx/dt
        print('vy: %.2f'%vy)
        #print('vx: %.2f'%vx)
        old_gray = gray_frame.copy()
        old_points = new_points
        x, y = new_points.ravel()
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    end = time.time()
    #print('Time Elapsed: %.2f seconds'%(end-start))
cap.release()
cv2.destroyAllWindows()
