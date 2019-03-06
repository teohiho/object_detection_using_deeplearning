# python flow --model cfg/yolo-voc-1.cfg --load bin/yolo-voc.weights --train --annotation annotations3 --dataset images3 --gpu 0.8

import cv2
from darkflow.net.build import TFNet
import time
import csv
import numpy as np

option = {
    # 'model': 'cfg/yolo.cfg',
    'model': 'cfg/yolo-voc-1.cfg',
    'load': 1739,
    # 'load': "weights/yolov2.weights",
    'threshold': 0.3,
    'gpu': 0.6
}
tfnet = TFNet(option)
# read video
capture = cv2.VideoCapture('video-01.mp4')

number_frame = 0

alpha = 2 
beta = 100

def box_large(tl, br):
    return br[0] - tl[0]
# Write file csv using dictwriter


while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    img = frame.copy()
    img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    person = 0
    results = []
    if ret:
        number_frame += 1
        if number_frame % 1 == 0:
            results = tfnet.return_predict(img)
            print(results)
            for result in results:
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                if box_large(tl, br) < 1000:
                    person += 1
                    frame = cv2.rectangle(frame, tl, br, (0, 255, 0), 1)
                    frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                    
        
        cv2.imshow("frame", frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
