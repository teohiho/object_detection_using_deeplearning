# python flow --model cfg/yolo-voc-1.cfg --load bin/yolo-voc.weights --train --annotation annotations3 --dataset images3 --gpu 0.8 --epoch 350
# Lấy dữ liệu để train từ ảnh
import cv2
from darkflow.net.build import TFNet
import os
from lxml import etree
import xml.etree.cElementTree as ET
import time
import matplotlib.pyplot as plt
#import imutils

option = {
    'model': 'cfg/yolo-voc-1.cfg',
    'load': 8000,
    'threshold': 0.75,
    'gpu': 0.5
}
tfnet = TFNet(option)

number_frame = 0
number_img = 0

tl_list = []
br_list = []
object_list = []
image_folder = 'images3'
savedir = 'annotations3'
obj = 'vehicle'


def write_xml(folder, image, image_name, objects, tl, br, savedir):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    height, width, depth = image.shape

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = image_name
    ET.SubElement(annotation, 'segmented').text = '0'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    for obj, topl, botr in zip(objects, tl, br):
        ob = ET.SubElement(annotation, 'object')
        ET.SubElement(ob, 'name').text = obj
        ET.SubElement(ob, 'pose').text = 'Unspecified'
        ET.SubElement(ob, 'truncated').text = '0'
        ET.SubElement(ob, 'difficult').text = '0'
        bbox = ET.SubElement(ob, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(topl[0])
        ET.SubElement(bbox, 'ymin').text = str(topl[1])
        ET.SubElement(bbox, 'xmax').text = str(botr[0])
        ET.SubElement(bbox, 'ymax').text = str(botr[1])

    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = os.path.join(savedir, image_name.replace('png', 'xml'))
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


def box_large(tl, br):
    return br[0] - tl[0]

file = open('result.txt', 'w')
for n, image_file in enumerate(os.scandir(image_folder)):
    if n < 437:
        continue
    print(image_file.name)
    image = cv2.imread(image_file.path)
    results = []
    results = tfnet.return_predict(image)
    for result in results:
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        if box_large(tl, br) < 250:
            frame = cv2.rectangle(image, tl, br, (0, 255, 0), 1)
            #frame = cv2.putText(image, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
            tl_list.append(tl)
            br_list.append(br)
            object_list.append(obj)
    cv2.imshow('img', frame)
    cv2.waitKey(0)
    write_xml(image_folder, image, image_file.name, object_list, tl_list, br_list, savedir)
    tl_list = []
    br_list = []
