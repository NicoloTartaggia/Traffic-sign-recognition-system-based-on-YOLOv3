from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

labels = ['uneven road', 'speed bump', 'slippery road', 'dangerous curve to the left',
          'dangerous curve to the right', 'double dangerous curve to the left', 'double dangerous curve to the right',
          'presence of children', 'bicycle crossing', 'cattle crossing', 'road works ahead',
          'traffic signals ahead', 'guarded railroad crossing', 'indefinite danger', 'road narrows',
          'road narrows from the left', 'road narrows from the right', 'priority at the next intersection',
          'intersection where the priority from the right is applicable', 'yield right of way',
          'yield to oncoming traffic', 'stop', 'no entry for all drivers', 'no bicycles allowed',
          'maximum weights allowed', 'no cargo vehicles allowed', 'maximum width allowed', 'maximum height allowed',
          'no traffic allowed in both directions', 'no left turn', 'no right turn', 'no passing to the left',
          'maximum speed limit', 'mandatory walk for pedestrians and bicycles', 'mandatory direction (ahead)',
          'mandatory direction (others)', 'mandatory direction (ahead and right)', 'mandatory traffic cycle',
          'mandatory bicycle path', 'shared path pedestrians/bicycle/mopeds', 'no parking', 'no waiting or parking',
          'no parking from 1st to the 15th of the month', 'no parking from 16th till the end of the month',
          'priority over oncoming traffic', 'parking allowed', 'parking for handicap only',
          'parking exclusively for cars', 'parking exclusively for trucks', 'parking exclusively for buses',
          'parking on sidewalk mandatory', 'beginning of residential area', 'end of residential area',
          'one way traffic',
          'dead end', 'end of road works', 'pedestrian crosswalk', 'bicycles/mopeds crossing', 'parking ahead',
          'speed bump', 'end of priority road', 'priority road']


def load_image(path):
    i = image.load_img(path, target_size=(32, 32))
    _org = cv2.imread(path)
    i = img_to_array(i)
    i = np.expand_dims(i, axis=0)

    return i, _org


def show_result(_org, _pred):
    _org = imutils.resize(_org, width=400)
    cv2.putText(_org, str(_pred[0]), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(_org, str(labels[_pred[0]]), (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite("./classify/result.png", _org)


def classify_image(image_path, model_path):
    img, org = load_image(image_path)
    classifier = load_model(model_path)
    pred = classifier.predict_classes(img)
    print(classifier.predict(img))
    print("Class: " + str(pred[0]))
    print("Class label: " + str(labels[pred[0]]))
    show_result(org, pred)
    print("The input image enriched with class number and label can be found in ./classify")
    return str(labels[pred[0]])


if __name__ == '__main__':
    image_path = "./additional_images/cropped.jpg"
    model_path = "./models/CNN1.h5"
    classify_image(image_path, model_path)
