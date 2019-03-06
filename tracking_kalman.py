# python flow --model cfg/yolo-voc-1.cfg --load bin/yolo-voc.weights --train --annotation annotations3 --dataset images3 --gpu 0.8

import cv2
from darkflow.net.build import TFNet
from kalman.sort import Sort
import time
import csv
import numpy as np

#create tracker using framework Sort
tracker=  Sort(use_dlib = False)

# test2.mp4
# corners1 = (400, 360)
# corners2 = (350, 720)
# corners4 = (680, 360)
# corners3 = (1000, 720)

# train3.mp4
corners1 = (470, 450)
corners2 = (400, 500)
corners3 = (950, 500)
corners4 = (950, 450)
def sign(corners1, corners2, x, y):
    return (corners2[1] - corners1[1])*(x - corners1[0]) - (corners2[0]- corners1[0])*(y - corners1[1])

# corners1 , corners4 am
# corners3, corners4 duong

option = {
    'model': "cfg/yolo-voc-1.cfg",
    'load': 8000,
    'threshold': 0.1,
    'gpu': 0.25
}
tfnet = TFNet(option)
# read video
capture = cv2.VideoCapture('video/train3.mp4')

number_frame = 0

def box_large(tl, br):
    return br[0] - tl[0]
# Write file csv using dictwriter


while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    results = []

    if ret:
        number_frame += 1
        if number_frame % 1 == 0:
            cv2.line(frame, corners1, corners2, (255,255,0), 2)
            cv2.line(frame, corners1, corners4, (255,255,0), 2)
            cv2.line(frame, corners2, corners3, (255,255,0), 2)
            cv2.line(frame, corners3, corners4, (255,255,0), 2)
            results = tfnet.return_predict(frame)
            trk = np.array([[0, 0, 0, 0]])
            for result in results:
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                check = sign(corners1, corners4, br[0], br[1])< 0 and sign(corners1, corners2, br[0], br[1]) > 0 and sign(corners2, corners3, br[0], br[1]) > 0 and sign(corners3, corners4, br[0], br[1]) > 0
                if check:
                    temp = np.array([[tl[0], tl[1], br[0], br[1]]])
                    trk = np.append(trk, temp, axis=0)
                # frame = cv2.rectangle(frame, tl, br, (0, 255, 0), 3)
                # frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)]
            print(trk)
            trk = np.delete(trk, 0, axis=0)
            if len(trk) > 0:
                trackers = tracker.update(trk, frame)
                print(trackers)
                for track in trackers:
                    cv2.rectangle(frame, (int(track[0]), int(track[1])),(int(track[2]), int(track[3])), (0, 255, 255), 3)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(track[4]), (int(track[0]),int(track[1])), font, 1,(255,255,255), 2, cv2.LINE_AA)

            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            cv2.imshow("frame", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
