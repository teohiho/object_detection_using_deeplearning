import cv2
from darkflow.net.build import TFNet
import os
from lxml import etree
import xml.etree.cElementTree as ET
import time
#import imutils

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.4,
    'gpu': 0.5
}
tfnet = TFNet(option)
# read video
capture = cv2.VideoCapture('1.mp4')
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# out = cv2.VideoWriter('output.avi', fourcc, 25.0, (720, 405))
# colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
number_frame = 0
number_img = 0

tl_list = []
br_list = []
object_list = []
image_folder = 'images'
savedir = 'annotations'
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


def line_select_callback(clk, rls):
    global tl_list
    global br_list
    global object_list
    tl_list.append((int(clk.xdata), int(clk.ydata)))
    br_list.append((int(rls.xdata), int(rls.ydata)))
    object_list.append(obj)


def onkeypress(event):
    global object_list
    global tl_list
    global br_list
    global img
    if event.key == 'q':
        print(object_list)
        write_xml(image_folder, frame, '{:06}.png'.format(number_img), object_list, tl_list, br_list, savedir)
        br_list = []
        object_list = []
        img = None


# Write file csv using dictwriter
file = open('result.txt', 'w')
while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    # frame = imutils.resize(frame, width=640)
    person = 0
    results = []
    if ret:
        number_frame += 1
        if number_frame % 20 == 0:
            number_img += 1
            results = tfnet.return_predict(frame)
            # print(results)
            for result in results:
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                if box_large(tl, br) < 250 and (label == 'motorbike'):
                    person += 1
                    # global tl_list
                    # global br_list
                    # global object_list 
                    fig, ax = plt.subplots(1, figsize=(14.5, 10))
                    mngr = plt.get_current_fig_manager()
                    # mngr.window.setGeometry(250, 40, 800, 600)
                    image = frame
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ax.imshow(image)
                    toggle_selector.RS = RectangleSelector(
                        ax, line_select_callback,
                        drawtype='box', useblit=True,
                        button=[1], minspanx=5, minspany=5,
                        spancoords='pixels', interactive=True,
                    )
                    bbox = plt.connect('key_press_event', toggle_selector)
                    key = plt.connect('key_press_event', onkeypress)
                    plt.tight_layout()
                    plt.show()
                    plt.close(fig)
                    # print(tl_list, br_list)

            cv2.imwrite('./images/' + '{:06}.png'.format(number_img), frame)
            print(number_frame, person, frame.shape[1], frame.shape[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        # out.release()
        cv2.destroyAllWindows()
        file.close()
        break
