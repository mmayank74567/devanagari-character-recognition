from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-tr", "--train", required=True,
	help="path to the trianing data")
ap.add_argument("-te", "--test", required=True,
	help="path to the testing data")
ap.add_argument("-trar", "--train_arr", required=True,
	help="output path to the training data array (include the file name you wish to get created)")
ap.add_argument("-trlar", "--train_label", required=True,
	help="output path to the training label array (include the file name you wish to get created)")
ap.add_argument("-tear", "--test_arr", required=True,
	help="output path to the training data array (include the file name you wish to get created)")
ap.add_argument("-telar", "--test_label", required=True,
	help="output path to the training label array (include the file name you wish to get created)")
args = vars(ap.parse_args())

#loading the training and testing data
train_data = []
train_labels = []
train_image_path = sorted(list(paths.list_images(args["train"])))
print('Total images in training set are', len(train_image_path))

test_data = []
test_labels = []
test_image_path = sorted(list(paths.list_images(args["test"])))
print('Total images in test set are', len(test_image_path))

#images to arrays
print('Converting training images into arrays')
for t in train_image_path:
  image = cv2.imread(t)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = img_to_array(image)
  image = np.uint8(image)
  train_data.append(image)
  label = t.split(os.path.sep)[-2]
  train_labels.append(label)
print('Done!') 


print('Storing the arrays')
np.save(args["train_arr"], train_data)
np.save(args["train_label"], train_labels)
print('Done!') 


print('Converting testing images into arrays')
for t in test_image_path:
    image = cv2.imread(t)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = img_to_array(image)
    image = np.uint8(image)
    test_data.append(image)
    test_label = t.split(os.path.sep)[-2]
    test_labels.append(test_label)
print('Done!') 

print('Storing the arrays')

np.save(args["test_arr"], test_data)
np.save(args["test_label"], test_labels)
print('Done!') 