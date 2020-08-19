from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2

labels = ['uneven road', 'speed bump', 'slippery road', 'dangerous curve to the left',
          'dangerous curve to the right', 'double dangerous curve to the left', 'double dangerous curve to the right',
          'presence of children', 'bicycle crossing', 'domestic animal crossing', 'wild animals crossing', 'road works ahead',
          'traffic signals ahead', 'guarded railroad crossing', 'indefinite danger', 'road narrows',
          'road narrows from the left', 'road narrows from the right', 'priority at the next intersection',
          'intersection where the priority from the right is applicable', 'yield right of way',
          'yield to oncoming traffic', 'stop', 'no entry for all drivers', 'no bicycles allowed',
          'maximum weights allowed', 'no cargo vehicles allowed', 'maximum width allowed', 'maximum height allowed',
          'no traffic allowed in both directions', 'no left turn', 'no right turn', 'no passing to the left',
          'maximum speed limit', 'mandatory walk for pedestrians and bicycles', 'mandatory direction (ahead)',
          'mandatory direction (right)', 'mandatory direction (keep right)', 'mandatory direction (ahead and right)',
          'mandatory direction (left)', 'mandatory direction (keep left)', 'mandatory direction (ahead and left)',
          'mandatory traffic cycle', 'mandatory bicycle path', 'shared path pedestrians/bicycle',
          'no parking', 'no waiting or parking', 'priority over oncoming traffic', 'one way traffic',
          'dead end', 'pedestrian crosswalk', 'bicycles crossing', 'parking area',
          'speed bump', 'end of priority road', 'priority road']


def load_image(img):
    i = cv2.resize(img, (32, 32))
    i = img_to_array(i)
    i = np.expand_dims(i, axis=0)

    return i


def load_imagetest(path):
    i = image.load_img(path, target_size=(32, 32))
    i = img_to_array(i)
    i = np.expand_dims(i, axis=0)

    return i


# def show_result(_org, _pred):
#    _org = imutils.resize(_org, width=400)
#    cv2.putText(_org, str(_pred[0]), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#    cv2.putText(_org, str(labels[_pred[0]]), (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#    cv2.imwrite("./classify/result.png", _org)


def classify_image(img_path, model_path):
    # img = load_image(img)
    img = load_imagetest(img_path)
    classifier = load_model(model_path)
    pred = classifier.predict_classes(img)
    pred_proba = classifier.predict(img)
    #print(pred_proba)
    #print(pred_proba.max())
    #print("Class: " + str(pred[0]))
    print("Class label: " + str(labels[pred[0]]))
    print("The input image enriched with class number and label can be found in ./classify")
    return str(labels[pred[0]]), pred_proba.max()


#model_path = "./models/CNN2.h5"
#img = cv2.imread(r"C:\Users\loren\Desktop\crop.jpg")
#print(classify_image(r"C:\Users\loren\Desktop\temp.jpg", model_path))
