import time
import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

def box_large(tl,br):
	return br[0]-tl[0]


options = {
	'model': 'cfg/yolo-voc-1.cfg',
    'load': 2400,
    'threshold': 0.3,
    'gpu': 0.25
}

tfnet = TFNet(options)
stime = time.time()
# read the color image and covert to RGB

img = cv2.imread('test4.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# use YOLO to predict the image
results = tfnet.return_predict(img)
print(len(results))
# pull out some info from the results

for result in results:
    tl = (result['topleft']['x'], result['topleft']['y'])
    br = (result['bottomright']['x'], result['bottomright']['y'])
    label = result['label']
    if box_large(tl, br) < 300:
        img = cv2.rectangle(img, tl, br, (0, 255, 0), 2)
        img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

cv2.imshow("frame", img)
print('FPS {:.1f}'.format(1 / (time.time() - stime)))
cv2.waitKey(0)
