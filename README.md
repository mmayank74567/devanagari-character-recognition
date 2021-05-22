# Devanagari Character Recognition using ResNet

## Introduction
This is the code for our submission to IEEE 9th International Conference on Reliability, Infocom Technologies and Optimization (ICRITO'2021). The work includes creating an image recognition model to classify characters written in Devanagari script. The model is based on the ResNet architecture and accomodates 85 convolution layers. This code can also be implemented to perform other image classification tasks by changing the hyper-parameters and design architecture according to need. 


## Dataset
We used the Devanagari Handwritten Character Dataset (DHCD) published in [Deep learning based large scale handwritten Devanagari character recognition](https://ieeexplore.ieee.org/document/7400041). The dataset can be downloaded from [here](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset). This dataset holds 91,000 grayscale images of 32x32 pixel dimensions across 46 different categories (36 consonants and 10 numeral classes).

### Loading the dataset
The `data_loading.py` is used to convert the images in the dataset into arrays and store the corresponding image label. To execute the script, run:
```shell
python data_loading.py --train [path to the training set] --test [path to testing set] --train_arr [path to store generated numpy array of training images] --train_label [path to store generated numpy array of training images' labels] --test_arr [path to store generated numpy array of testing images] --test_label [path to store generated numpy array of testing images' labels]
```

Ensure to include the name of the `.npy` file that you wish to be created after running the script. A sample command line argument is demonstrated below:


```shell
python data_loading.py --train DevanagariHandwrittenCharacterDataset/Train --test DevanagariHandwrittenCharacterDataset/Test --train_arr output/train_array.npy --train_label output/train_label.npy --test_arr output/test_array.npy --test_label output/test_label.npy
```
After successfully running the code, there will be four `.npy` files created in your specified folder. (The name of the folder and files are subject to the input given by the user in the previous argument)
```shell
├── output
   ├── train_array.npy
   ├── train_label.npy
   ├── test_array.npy
   └── test_label.npy
```

## Structure of the directory
The structure of the directory is represented below:
```shell

├── callbacks
│   ├── __init__.py
│   ├── epochcheckpoint.py
│   ├── learningratedecay.py
│   └── trainingmonitor.py
├── nn
│   ├── __init__.py
│   ├── resnet.py
├── output
|   ├── train_array.npy
|   ├── train_label.npy
|   ├── test_array.npy
|   └── test_label.npy
├── checkpoints
└── train.py

```
### Callbacks
* The `TrainingMonitor` callback (in `trainingmonitor.py`) is called after every epoch during training. It serializes the loss and accuracy of both training and testing set to the the disk (as a JSON file) and makes a loss vs accuracy plot of the data. The output of this callback is essential to keep a track and monitor the training.
* The `EpochCheckpoint` callback (in `epochcheckpoint.py`) serializes the entire model to the disk after every specified number of epochs (default = 5). This enables resuming training from a specific model checkpoint in the event of any interruption.  
* The `learningratedecay.py` defines a custom learning rate function which is automatically applied during the training using the `LearningRateScheduler` class of the Keras library.

### ResNet