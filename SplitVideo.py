import cv2
import os

mypath = './images/'
if not os.path.isdir(mypath):
   os.makedirs(mypath)
capture = cv2.VideoCapture('./data/video/test1.mp4')
number_frame = 0
number_img = 0
while (capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        number_frame += 1
        if number_frame % 26 == 0:
            number_img += 1
            cv2.imwrite('./images/' + '{:06}.png'.format(number_img), frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or number_frame == 100:
        break
