# python flow --model cfg/yolo-voc-1.cfg --load bin/yolo-voc.weights --train --annotation annotations3 --dataset images3 --gpu 0.8

import cv2
from darkflow.net.build import TFNet
from kalman.sort import Sort
import time
import csv
import numpy as np
from math import sqrt
import imutils


class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(old_frame, (x, y), 3, (255, 0, 0), -1)
            self.points.append((x, y))

    def setROI(self):
        corners1 = self.points[0]
        corners2 = self.points[1]
        corners3 = (self.points[2][0], corners2[1])
        corners4 = (self.points[3][0], corners1[1])
        threshold_detection = self.points[4][1]
        threshold_delete = self.points[5][1]
        return corners1, corners2, corners3, corners4, threshold_detection, threshold_delete


def return_box(r1, r2, r3, r4, r6, r7):
    cor1 = (r[0], r[1])
    cor2 = (r[0], r[1] + r[3])
    cor3 = (r[0] + r[2], r[1] + r[3])
    cor4 = (r[0] + r[2], r[1])
    return cor1, cor2, cor3, cor4


def draw_ROI(frame):
    cv2.line(frame, corners1, corners2, (255, 255, 0), 2)
    cv2.line(frame, corners1, corners4, (255, 255, 0), 2)
    cv2.line(frame, corners2, corners3, (255, 255, 0), 2)
    cv2.line(frame, corners3, corners4, (255, 255, 0), 2)


def box_large(tl, br):
    return br[0] - tl[0]


def point_center(tl, br):
    return ((tl[0] + br[0])/2, (tl[1]+br[1])/2 + 0.25*(br[1]-tl[1]))
    # return ((tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2)


def sign(corners1, corners2, x, y):
    return (corners2[1] - corners1[1]) * (x - corners1[0]) - (corners2[0] - corners1[0]) * (y - corners1[1])


def distance(point1, point2):
    return sqrt((point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1]))


# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))


option = {
    'model': "cfg/yolo-voc-1.cfg",
    'load': 8000,
    'threshold': 0.5,
    'gpu': 0.5
}
tfnet = TFNet(option)

# read video
capture = cv2.VideoCapture('data/video/nguyenvantroi.mp4')
ret, old_frame = capture.read()
old_frame = imutils.resize(old_frame, width=1280)
coordinateStore1 = CoordinateStore()
cv2.namedWindow('image')
cv2.setMouseCallback('image', coordinateStore1.select_point)
while(1):
    cv2.imshow('image', old_frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(coordinateStore1.points) > 5:
        break
cv2.destroyAllWindows()

corners1, corners2, corners3, corners4, threshold_detection, threshold_delete = coordinateStore1.setROI()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = np.array([[[0, 0]]], dtype=np.float32)
mask = np.zeros_like(old_frame)
number_frame = 0
object1 = 0
bike = 0
capacity = 0
numner_object = np.array([0])

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('data/video/output.mp4', fourcc, 14, (1280, 720))


while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    frame = imutils.resize(frame, width=1280)
    results = []
    if ret:
        if number_frame % 1 == 0:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # get update the vector!
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1
            good_old = p0
            number_frame += 1

            # draw the ROI
            draw_ROI(frame)

            # predict and return object is detected
            results = tfnet.return_predict(frame)
            for result in results:
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                point_flow = point_center(tl, br)

                # check the point_flow in ROI!
                check = sign(corners1, corners4, point_flow[0], point_flow[1]) < 0 and sign(corners3, corners4, point_flow[0], point_flow[1]) > 0 and sign(
                    corners1, corners2, point_flow[0], point_flow[1]) > 0 and sign(corners2, corners3, point_flow[0], point_flow[1]) > 0 and point_flow[1] <= threshold_detection
                if check and label != "" and box_large(tl, br) < 250:
                    frame = cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                    chk = False
                    for i in p1:
                        # distance between point_flow and all point in p1
                        print(distance(point_flow, (i[0][0], i[0][1])))
                        if distance(point_flow, (i[0][0], i[0][1])) < 50:
                            chk = True
                    if not chk:
                        p2 = np.array(
                            [[[point_flow[0], point_flow[1]]]], dtype=np.float32)
                        p1 = np.append(p1, p2, axis=0)
                        bike += 1
                        object1 += 1
                        numner_object = np.append(numner_object, object1)

                size_of_p1 = len(p1)
                index_cur = 0
                while index_cur < size_of_p1:
                    if int(p1[index_cur][0][1]) >= threshold_delete:
                        p1 = np.delete(p1, index_cur, axis=0)
                        size_of_p1 -= 1
                        numner_object = np.delete(numner_object, index_cur)
                    else:
                        index_cur += 1

            # draw the point object is dectected
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
                # frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
                if i < len(numner_object):
                    cv2.putText(frame, str(
                        numner_object[i]), (c, d), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), thickness=3)

            cv2.putText(frame, str(capacity) + "xe/phut", (int(frame.shape[0] / 2), int(
                frame.shape[1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(200, 50, 75), thickness=3)
            # img = cv2.add(frame, mask)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            cv2.imshow("frame", frame)
            # out.write(frame)

            # Update old_fray and p0 for next flow1
            old_gray = frame_gray.copy()
            p0 = p1.reshape(-1, 1, 2)

            if len(p0) > 80:
                p0 = np.array([[[0, 0]]], dtype=np.float32)
            if number_frame % 840 == 0:
                capacity = bike
                bike = 0

            # for two second reset mask
            # if number_frame%30 == 0:
                # mask = np.zeros_like(old_frame)
        # press "q" to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
out.release()
capture.release()
cv2.destroyAllWindows()
