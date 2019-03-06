import time
import cv2
from darkflow.net.build import TFNet
import numpy as np

def box_large(tl,br):
	return br[0]-tl[0]


options = {
    # 'model': 'cfg/yolo.cfg',
    'model': 'cfg/yolo-voc-1.cfg',
    'load': 1250,
    # 'load': "weights/yolov2.weights",
    'threshold': 0.2,
    'gpu': 0.6
}

alpha = 2 
beta = 100


tfnet = TFNet(options)
stime = time.time()
# read the color image and covert to RGB
cap = cv2.VideoCapture('video-01.mp4')
ret, img = cap.read()
img_brighter = img.copy()
#img = cv2.imread('video/index.jpeg', cv2.IMREAD_COLOR)
brighter = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
img_brighter = cv2.cvtColor(brighter, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# use YOLO to predict the image
results = tfnet.return_predict(img)
print(len(results))
results_brighter = tfnet.return_predict(img_brighter)
print(len(results_brighter))
# pull out some info from the results

for result in results:
    tl = (result['topleft']['x'], result['topleft']['y'])
    br = (result['bottomright']['x'], result['bottomright']['y'])
    label = result['label']
    if box_large(tl, br) < 300:
        img = cv2.rectangle(img, tl, br, (0, 255, 0), 2)
        img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

for result in results_brighter:
    tl = (result['topleft']['x'], result['topleft']['y'])
    br = (result['bottomright']['x'], result['bottomright']['y'])
    label = result['label']
    if box_large(tl, br) < 300:
        img_brighter = cv2.rectangle(img_brighter, tl, br, (0, 255, 0), 2)
        img_brighter = cv2.putText(img_brighter, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
print('FPS {:.1f}'.format(1 / (time.time() - stime)))
cv2.imshow("img", img)
cv2.imshow("img_brighter", img_brighter)
cv2.waitKey(0)
cv2.destroyAllWindows()









