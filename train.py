from callbacks.learningratedecay import PolynomialDecay
from callbacks.epochcheckpoint import EpochCheckpoint
from callbacks.trainingmonitor import TrainingMonitor
from nn.resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import BaseLogger
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
import json
import numpy as np
import sys
import os



apo = argparse.ArgumentParser()
apo.add_argument("-n","--npy",required = "True", help = "Path to the folder with the .npy files")
apo.add_argument("-o","--output",required = "True", help = "Path to the output folder")
apo.add_argument("-e","--epochs",required = "True",type=int, help = "Number of epochs you wish to train the model")
apo.add_argument("-lr","--lrate",required = "True",type=float, help = "Initial learning rate for the training")
apo.add_argument("-c","--checkpoint",required = "True", help = "Path to (checkpoints) folder to store the model checkpoints during training")

args = vars(apo.parse_args())

#path to the npy files (change if file names different)
train_data_path = os.path.sep.join([args["output"], "train_array.npy"])
train_label_path = os.path.sep.join([args["output"], "train_label.npy"])
test_data_path = os.path.sep.join([args["output"], "test_array.npy"])
test_label_path = os.path.sep.join([args["output"], "test_label.npy"])

trainX = np.load(train_data_path)
trainY = np.load(train_label_path)
testX = np.load(test_data_path)
testY = np.load(test_label_path)

print("[INFORMATION] The trainX shape ", trainX.shape)
print("[INFORMATION] The trainY shape ",trainY.shape)
print("[INFORMATION] The testX shape ",testX.shape)
print("[INFORMATION] The testY shape ",testY.shape)

#learning rate decay (as linear lrdecay, thus the power = 1)
schedule = PolynomialDecay(maxEpochs=args["epochs"], power=1,baseLr = args["lrate"])

#if training for the start, set the following. else if loading from a checkpoint
#mention the model path and epoch number where to start training from again
model = None
startepoch = 0

trainX = trainX.astype("float")
testX = testX.astype("float")
#  mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean
# labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# image generator for data augmentation
# modify operations according to requirements
aug = ImageDataGenerator(width_shift_range=0.1,
	height_shift_range=0.1,
	fill_mode="nearest")#replaces the empty area with the nearest pixel values

#if training from starting
if model is None:
	print("[INFORMATION] Compiling the model.")
	opt = SGD(lr=args["lrate"])
	model = ResNet.build(32, 32, 1, 46, (9, 9, 9),
		(64, 64, 128, 256), reg=0.0005)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])
# if loading a model from a checkpoint
else:
	print("[INFORMATION] Loading the model- {}".format(model))
	model = load_model(model)

	# update the learning rate
	print("[INFORMATION] old learning rate: {}".format(
		K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-2)
	print("[INFORMATION] new learning rate: {}".format(
		K.get_value(model.optimizer.lr)))

#storing a plot of the model on the disk
model_plot_path = os.path.sep.join([args["output"], "model_path.png"])
plot_model(model, to_file=model_plot_path, show_shapes=True, show_layer_names=True)

#the path to fig and json that will be updated and stored during the course of the training
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])

#the callbacks
callbacks = [
	EpochCheckpoint(args["checkpoint"], every=5,
		startAt=startepoch),
	TrainingMonitor(figPath,jsonPath=jsonPath,startAt=startepoch),LearningRateScheduler(schedule)]

# training the network
print("[INFORMATION] Training the network")
model.fit_generator(
	aug.flow(trainX, trainY, batch_size=128),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // 128, epochs=args["epochs"],
	callbacks=callbacks, verbose=1)