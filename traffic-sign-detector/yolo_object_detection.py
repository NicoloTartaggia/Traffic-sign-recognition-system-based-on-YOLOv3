import glob
import time
import cv2

from ClassifyImages import *


""" ------------------------- NET BUILDING ------------------------- """
weights_path = "./weights/yolov3_training_final.weights"
cfg_path = "./cfg/yolov3_testing.cfg"

net = cv2.dnn.readNet(weights_path, cfg_path)  # build net with cv2
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
""" ---------------------------------------------------------------- """

# Model path
model_path = "./models/CNN2.h5"


def traffic_sign_detector(img):
    height, width, _ = img.shape

    """ ------------------ Detecting objects ------------------ """
    # The cv2.dnn.blobFromImage  function returns a blob  which is our input image after mean subtraction,
    # normalizing, and channel swapping
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    # Storing detections made by forward procedure
    outs = net.forward(output_layers)
    """ ------------------------------------------------------- """

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
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
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            crop_img = img[y:y+h, x:x+w]
            cv2.imwrite("./classify/temp.jpg", crop_img)
            label, prob = classify_image("./classify/temp.jpg", model_path)
            total_confidence = prob * confidences[i]
            color = [0, 255, 255]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
            if label == "stop":
                cv2.rectangle(img, (x, y + 3), (x + (len(label) * 22), y - 10), color, -1)
            else:
                cv2.rectangle(img, (x, y + 3), (x + (len(label) * 12), y - 10), color, -1)
            cv2.putText(img, label + " " + str('%.0f' % (total_confidence * 100)) + "%", (x, y), font, 1, (0, 0, 0), 2)
    return img


# compute the detector accuracy
def test_detector_accuracy(path):
    txt_files = glob.glob(path + "/*.txt")
    txt_cont = 0
    tot_cont = 0
    true_cont = 0
    conf = 0.1
    start = time.time()
    for img_path in glob.glob(path + "/*.jpg"):
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        f = open(txt_files[txt_cont], "r")
        lines = f.readlines()
        tot_cont += len(lines)

        """ ------------------ Detecting objects ------------------ """
        # The cv2.dnn.blobFromImage  function returns a blob  which is our input image after mean subtraction,
        # normalizing, and channel swapping
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        # Storing detections made by forward procedure
        outs = net.forward(output_layers)
        """ ------------------------------------------------------- """

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
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

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                x, y, w, h = x/width, y/height, w/width, h/height
                for line in lines:
                    line = line.split()
                    x_diff = abs(x - float(line[1]))
                    y_diff = abs(y - float(line[2]))
                    w_diff = abs(w - float(line[3]))
                    h_diff = abs(h - float(line[4]))

                    if x_diff <= conf and y_diff <= conf and w_diff <= conf and h_diff <= conf:
                        true_cont += 1

        txt_cont += 1
    time_spent = time.time() - start
    return true_cont / tot_cont, time_spent


def image_analysis(image_path):
    for img_path in glob.glob(image_path):
        # Loading image
        img = cv2.imread(img_path)
        img = traffic_sign_detector(img)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_analysis(video_path, flag):  # flag 1 for real-time video analysis
    if flag == 1:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('output.avi', fourcc, 20, size)

    # loop through all the frames
    while True:
        _, img = cap.read()
        height, width, _ = img.shape
        img = traffic_sign_detector(img)
        writer.write(img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # image_analysis(r"C:\Users\loren\Desktop\train\*.jpg")
    video_analysis("day.mp4", 0)
    # print(test_detector_accuracy(r"C:\Users\loren\Desktop\validation_set"))
