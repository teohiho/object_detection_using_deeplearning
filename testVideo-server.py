# python flow --model cfg/yolo-voc-1.cfg --load bin/yolo-voc.weights --train --annotation annotations3 --dataset images3 --gpu 0.8

import cv2
from darkflow.net.build import TFNet
from stream_server import StreamingSever
import time

#import csv

option = {
    'model': 'cfg/yolo-voc-1.cfg',
    'load': 8000,
    'threshold': 0.2,
    'gpu': 0.25
}
tfnet = TFNet(option)
# read video
capture = cv2.VideoCapture('video/test.mp4')



def box_large(tl, br):
    return br[0] - tl[0]
# Write file csv using dictwriter
def gen_gray_image():
    number_frame = 0
    while (capture.isOpened()):
        stime = time.time()
        ret, frame = capture.read()
        person = 0
        results = []
        if ret:

            number_frame += 1
            if number_frame % 1 == 0:
                results = tfnet.return_predict(frame)
                # print(results)
                for result in results:
                    tl = (result['topleft']['x'], result['topleft']['y'])
                    br = (result['bottomright']['x'], result['bottomright']['y'])
                    label = result['label']
                    if box_large(tl, br) < 300:
                        person += 1

                        frame = cv2.rectangle(frame, tl, br, (0, 255, 0), 1)
                        #frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                        
                print('FPS {:.1f}'.format(1 / (time.time() - stime)))
                yield frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            capture.release()
            cv2.destroyAllWindows()
            break
app = StreamingSever(gen_gray_image)
app.run(host="0.0.0.0", port="8000", debug=True)