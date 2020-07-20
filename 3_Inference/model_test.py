import keras
import tensorflow as tf
import os
from keras.models import load_model, Model
import cv2

root = r"/Users/serafinakamp/Desktop/YOLO_test/TrainYourOwnYOLO/Data"

weights = os.path.join(root,"Model_Weights/trained_weights_final.h5")

test_images = os.listdir(os.path.join(root,"Source_Images/Test_Images"))

model = load_model(weights)

image = cv2.imread(os.path.join(root,"Source_Images/Test_Images",test_images[0]))

image = cv2.resize(image,(416,416))
image = np.expand_dims(image,0)

print(np.shape(image))
