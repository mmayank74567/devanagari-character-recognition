# Devanagari Character Recognition using ResNet

## Introduction
This is the code for an image recognition model to classify characters written in Devanagari script. The model is based on the ResNet architecture and accomodates 85 convolution layers. This code can also be implemented to perform other image classification tasks by changing the hyper-parameters and design architecture according to need. 

## Installation
* Download Anaconda or Miniconda distribution from their [website](https://www.anaconda.com/products/individual).
* Clone this repository
```shell
git clone https://github.com/mmayank74567/devanagari-character-recognition.git
```
* Create an environment and install the dependencies using the `env.yml` file.
```shell
conda env create -f env.yml
```
An environment by the name `dev_env` will be created. Run the following command to ensure that an environment with this name has been successfully created.
```shell
conda env list
```
* Activate the environment to run the code using the installed dependencies.
```shell
conda activate dev_env
#change directory 
cd devanagari-character-recognition/
```
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
The ResNet architecture is implemented in the `resnet.py` inside the `nn` module. The bottleneck and pre-activation version of the ResNet is implemented. 

The `residual_module` function within the `ResNet` class defines the structure of a residual module. Three `CONV` blocks with `K/4` filters for first and second block and `K` filters for the third block are stacked. Each `CONV` block holds a series of `BatchNormalization -> ReLU -> Conv2D` layers. 

The residual blocks from the `residual_module` function are stacked together using the `build` function as per the desired requirements. The `stride` and `filters` parameters of the function are given as <i>lists</i>. `stride[i]` element describes the number of residual modules in the <em>i <sup>th</sup></em>  set of the network architectures. `filter[i+1]` defines the number of filters `K` used by all the `CONV` blocks in the <em>i <sup>th</sup></em> set. Spatial dimensions are reduced when moving from set <em>i</em> to set <em>i+1</em>.

The network architecture starts with a batch normalization layer. It is followed by a ` CONV ` layer with a total of `filter[0]` filters. After these two initial layers, all the sets of residual modules are stacked together. At last, average pooling and the softmax classifier is applied. 

### Training
Deploy the `train.py` file to initiate the training process. To execute the script, run --
```shell
python train.py --npy [path to the folder with the .npy files] --output [path to the output folder] --epochs [Number of epochs you wish to train the model] --lrate [Initial learning rate for the training] --checkpoint [Path to (checkpoints) folder to store the model checkpoints during training]]
```

A sample command line argument is demonstrated below:
```shell
python train.py --output output --npy output --epochs 50 --lrate 0.01 --checkpoint checkpoints
```

In our implementation, we have initialized the `stride` and `filters` arguments as (9, 9, 9) and (64, 64, 128, 256) respectively. The first `CONV` layer (before the residual module stacking), learns `filter[0]` number of filters, i.e. 64. The first set contains `stride[0]`, i.e. 9 residual modules. These 9 residual modules in the first set learn `filter[1]` number of filters, i.e. 64. SImilarly, the second set has 9 residual modules, each learning 128 filters and the third set has 9 residual modules, with 256 filters applied to each. The plot of our network architecture is shown in `model.png` file. 

While training, the `JSON` file and a loss vs accuracy plot will be stored in the folder provided within the `--output` argument after every epoch. Model checkpoints will be stored in the folder provided within the `--checkpoint` argument after every 5 epochs (or the number set by the user). In the event of an interruption, these checkpoints can be used to restart training from a specific point in the history by mentioning the model path and the epoch number to start training from again in the `train.py`file.

## Acknowledgement
* The code implementation is inspired by the resources provided on the [PyImageSearch](https://www.pyimagesearch.com/) website.

## Citation
```
@INPROCEEDINGS{9582192,
  author={Mishra, Mayank and Choudhury, Tanupriya and Sarkar, Tanmay},
  booktitle={2021 IEEE India Council International Subsections Conference (INDISCON)}, 
  title={Devanagari Handwritten Character Recognition}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/INDISCON53343.2021.9582192}}
  ```




