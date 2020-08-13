import cv2
import numpy as np
import glob
from PIL import Image, ImageDraw, ImageFont
import random

# Load Yolo
# from cv2 import VideoCapture

weights_path = "./weights/yolov3.weights"
cfg_path = "./cfg/yolov3.cfg"
net = cv2.dnn.readNet(weights_path, cfg_path)

# Name custom object
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

#classes = ["traffic_sign"]

# Images path
images_path = glob.glob(r"C:\Users\loren\Desktop\prova\*.jpg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# loop through all the images
cap = cv2.VideoCapture(0) #uncomment for webcam object detection
#cap = cv2.VideoCapture("video.mp4")
"""width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)"""

#for img_path in images_path: #uncomment for img object detection
while True:
    # Loading image
    #img = cv2.imread(img_path)
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    _, img = cap.read()
    height, width, _ = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    conf = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            conf = str('%.0f' % (confidences[i] * 100))
            color = colors[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + conf + "%", (x, y), font, 1, color, 2)

    # out.write(img)
    cv2.imshow("Image", img)
    #key = cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
# out.release()
cv2.destroyAllWindows()
