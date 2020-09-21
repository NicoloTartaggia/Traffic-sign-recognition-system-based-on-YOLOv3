import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras import metrics
from keras.models import load_model
import numpy as np

labels = ['uneven road', 'speed bump', 'slippery road', 'dangerous curve to the left',
          'dangerous curve to the right', 'double dangerous curve to the left', 'double dangerous curve to the right',
          'presence of children', 'bicycle crossing', 'domestic animal crossing', 'road works ahead',
          'traffic signals ahead', 'guarded railroad crossing', 'indefinite danger', 'road narrows',
          'road narrows from the left', 'road narrows from the right', 'priority at the next intersection',
          'intersection where the priority from the right is applicable', 'yield right of way',
          'yield to oncoming traffic', 'stop', 'no entry for all drivers', 'no bicycles allowed',
          'maximum weights allowed', 'no cargo vehicles allowed', 'maximum width allowed', 'maximum height allowed',
          'no traffic allowed in both directions', 'no left turn', 'no right turn', 'no passing to the left',
          'maximum speed limit', 'mandatory walk for pedestrians and bicycles', 'mandatory direction (ahead)',
          'mandatory direction (right)', 'mandatory direction (ahead and right)', 'mandatory direction (keep right)',
          'mandatory direction (left)', 'mandatory direction (ahead and left)', 'mandatory direction (keep left)',
          'mandatory traffic cycle', 'mandatory bicycle path', 'shared path pedestrians/bicycle',
          'no parking', 'no waiting or parking', 'priority over oncoming traffic', 'wild animals crossing',
          'one way traffic', 'dead end', 'pedestrian crosswalk', 'bicycles crossing', 'parking area',
          'speed bump', 'end of priority road', 'priority road']


def load_image(path):
    i = image.load_img(path, target_size=(32, 32))
    i = img_to_array(i)
    i = np.expand_dims(i, axis=0)
    return i


def classify_image(img_path, model_path):
    img = load_image(img_path)
    classifier = load_model(model_path, compile=False)
    pred = classifier.predict_classes(img)
    pred_proba = classifier.predict(img)
    return str(labels[pred[0]]), pred_proba.max()
