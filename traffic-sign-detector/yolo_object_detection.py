import glob
# from PIL import Image, ImageDraw, ImageFont
# import random
from ClassifyImages import *

# Load Yolo
# from cv2 import VideoCapture

""" -------------------------NET BUILDING ------------------------- """
weights_path = "./weights/yolov3_training_last.weights"  # https://drive.google.com/drive/u/0/folders/1TeorKkxJUxaWaTnhHe-_XSxtLiDDktiP
cfg_path = "./cfg/yolov3_testing.cfg"
net = cv2.dnn.readNet(weights_path, cfg_path)  # build net with cv2
""" --------------------------------------------------------------- """

# Model path
model_path = "./models/CNN3.h5"
# Images path
images_path = glob.glob(r"../../../../../../Scaricati/images/*.jpg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

"""cap = cv2.VideoCapture(0)  # uncomment for webcam object detection
cap = cv2.VideoCapture("video.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)"""

# loop through all the images
for img_path in images_path: # uncomment for img object detection
    # while True: # For video detection
        # Loading image
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        # _, img = cap.read()
        img = cv2.add(img, np.array([50.0]))
        height, width, _ = img.shape

        """ ------------------ Detecting objects ------------------ """
        # The cv2.dnn.blobFromImage  function returns a blob  which is our input image after mean subtraction,
        # normalizing, and channel swapping.
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        # Storing detections made by forward procedure
        outs = net.forward(output_layers)
        """ ------------------------------------------------------- """

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
                crop_img = img[y:y + h, x:x + w]
                label = classify_image(crop_img, model_path)
                conf = str('%.0f' % (confidences[i] * 100))
                color = [0, 0, 0]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + conf + "%", (x, y), font, 1, color, 2)

        # out.write(img)
        cv2.imshow("Image", img)
        key = cv2.waitKey(0)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

# cap.release()
# out.release()
cv2.destroyAllWindows()
