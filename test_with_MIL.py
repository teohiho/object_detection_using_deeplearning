# python flow --model cfg/yolo-voc-1.cfg --load bin/yolo-voc.weights --train --annotation annotations3 --dataset images3 --gpu 0.8

import cv2
from darkflow.net.build import TFNet
from kalman.sort import Sort
import time
import csv
import numpy as np
from math import sqrt
import imutils


def return_box(r):
    cor1 = (r[0], r[1])
    cor2 = (r[0], r[1] + r[3])
    cor3 = (r[0] + r[2],r[1] + r[3])
    cor4 = (r[0] + r[2], r[1])
    return cor1, cor2, cor3, cor4


def draw_ROI(frame):
    cv2.line(frame, corners1, corners2, (255,255,0), 2)
    cv2.line(frame, corners1, corners4, (255,255,0), 2)
    cv2.line(frame, corners2, corners3, (255,255,0), 2)
    cv2.line(frame, corners3, corners4, (255,255,0), 2)


def box_large(tl, br):
    return br[0] - tl[0]


def point_center(tl, br):
    #return ((tl[0] + br[0])/2, (tl[1]+br[1])/2 - 0.25*(br[1]-tl[1]))
    return ((tl[0] + br[0])/2, (tl[1]+br[1])/2)


def sign(corners1, corners2, x, y):
    return (corners2[1] - corners1[1])*(x - corners1[0]) - (corners2[0]- corners1[0])*(y - corners1[1])


def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

# update new box 
def update_with_iou(number_current, boxes_new, boxes_old, threshold = 0.5):
    check = False
    for new in boxes_new:
        chk = False
        for old in boxes_old:
            if iou(new, old) > threshold:
                chk = True
                break
        if not chk:
            boxes_old = np.append(boxes_old, np.array([new]), axis=0)
            check = True
            number_current+=1
    return check, number_current, boxes_old


#delete the box if it's not in ROI
def delete_box_not_in_ROI(boxes, threshold):
    cur = 0
    size_of_boxes = len(boxes)
    check = False
    while cur < size_of_boxes:
        if boxes[cur][1] > threshold:
            boxes = np.delete(boxes, cur, axis=0)
            size_of_boxes-=1
            check = True
        else: cur+=1
    return boxes, check


# format box = (x,y,w,h)
def create_new_multi_tracker(boxes, frame):
    tracker = cv2.MultiTracker_create()
    for box in boxes:
        ok = tracker.add(cv2.TrackerMIL_create(), frame, box)

    return tracker

def return_boxes_Yolo(results):
    boxes = np.array([[0,0,0,0]])
    for result in results:
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        box = np.array([[ tl[0], tl[1], br[0], br[1]]])
        boxes = np.append(boxes, box, axis=0)
    if len(boxes)>1:
        boxes = np.delete(boxes, 0, axis=0)
    return boxes


def convert_to_class_tuple(boxes):
    results = [tuple([0,0,10,10])]
    for box in boxes:
        results.append(tuple(box))
    if len(results) > 1:
        results.remove(results[0])
    return results


def convert_wh_to_xy(results):
    boxes = np.array([[0,0,0,0]])
    for result in results:
        box = np.array([[ result[0], result[1], result[0]+result[2], result[1]+result[3]]])
        boxes = np.append(boxes, box, axis=0)
    if len(boxes)>1:
        boxes = np.delete(boxes, 0, axis=0)
    return boxes


def convert_xy_to_wh(results):
    boxes = np.array([[0,0,0,0]])
    for result in results:
        box = np.array([[ result[0], result[1], -result[0]+result[2], -result[1]+result[3]]])
        boxes = np.append(boxes, box, axis=0)
    if len(boxes)>1:
        boxes = np.delete(boxes, 0, axis=0)
    return boxes
# Parameters for lucas kanade optical flow
color = np.random.randint(0,255,(100,3))

option = {
    'model': "cfg/yolo-voc-1.cfg",
    'load': 8000,
    'threshold': 0.2,
    'gpu': 0.8
}
tfnet = TFNet(option)


# read video
capture = cv2.VideoCapture('data/video/nguyenvantroi.mp4')
ret, first_frame = capture.read()
# Create first multi_tracker
boxes = [tuple([0,0,10,10])]
# tracker = create_new_multi_tracker(boxes, first_frame)
tracker = cv2.TrackerMIL_create()
ok = tracker.init(first_frame, boxes[0])
# set ROI for detection
# Roi = cv2.selectROI(first_frame, False)
# corners1, corners2, corners3, corners4 = return_box(Roi)


# parameter for saving video
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('data/video/output.mp4',fourcc, 14, (1280,720))

number_current=0
number_frame=0
while (capture.isOpened()):
    
    ret, frame = capture.read()
    results = []
    if ret:
        if number_frame % 1 == 0:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            number_frame += 1
            # draw the ROI 
            # draw_ROI(frame)
            stime = time.time()
            ok, boxes = tracker.update(frame)

            # results = tfnet.return_predict(frame)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            # # get box from results of yolo format (x1,y1,x2,y2)
            # boxes_yolo = return_boxes_Yolo(results)
            # boxes = convert_wh_to_xy(boxes)
            # check_update, number_current, boxes = update_with_iou(number_current, boxes_yolo, boxes)
            # boxes, check_delete = delete_box_not_in_ROI(boxes, threshold = 450)
            # boxes = convert_xy_to_wh(boxes)
            # if check_update == True or check_delete == True:
            #     boxes = convert_to_class_tuple(boxes)
            #     tracker = create_new_multi_tracker(boxes, frame)

            # for box in boxes:
            #     cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), (255,255,0))

            # cv2.rectangle(frame, (boxes[0],boxes[1]), (boxes[0]+boxes[2],(boxes[1]+boxes[3])), (255,255,0), 2)

            #img = cv2.add(frame, mask) 
            
            cv2.imshow("frame", frame)
            #out.write(frame)
        # press "q" to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):   
            break
    else:
        break
out.release()
capture.release()
cv2.destroyAllWindows()
