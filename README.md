# mobilenetv1-cifar10

This is a keras implementation for CIFAR-10 classification using reduced MobileNet-V1.

**training.py** file contains the full implementation of the model training pipeline.
**model.h5** is the best model.
**training_log.csv** contains loss and accuracy documentation for each epoch.
**plot.png** is the vizualization of the accuracy progress during the training, as follows:

![plot](/plot.png)

**CIFAR-10** is a dataset of a 32x32 colour images of 10 classes, which includes 50,000 training images and 10,000 test images.

The main idea behind **MobileNets** is to replace standart convolutions with depthwise separable convolutions in order to reduce calculations.
The network has 28 layers, the baseline network contains 3.23 million parameters. 
Attached is the MobileNet body architecture as it apppears in the papaer "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications":

![arch](/architecture.png)


MobiLeNet has two hyperparameters that trade off between size and accuracy of the network:
**width multiplier** shrinks the number of channels and thin the netwrok at each layer, and
**resolution multiplier** reduces the image resolution. Originally, both parameteres are in the range (0, 1].
In this project, instead of using resolution multiplier, **depth multiplier** is used in order to increase accuracy, as described in paper "An Enhanced Hybrid MobileNet".
The proposed depth multiplier values are 1, 2, and 4. In this project, alpha = 0.25 and delta = 4, this solution has 0.85 millions parameters.

