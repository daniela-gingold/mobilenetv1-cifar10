# mobilenetv1-cifar10

This is a keras implementation for CIFAR-10 classification using reduced MobileNet-V1.

**training.py** file contains the full implementation of the model training pipeline.
**model.h5** is the best model.
**training_log.csv** contains loss and accuracy documentation for each epoch.
**plot.png** is the vizualization of the loss and accuracy progress during the training, as follows:

![plot](/plot.png)

**CIFAR-10** is a dataset of a 32x32 colour images of 10 classes, which includes 50,000 training images and 10,000 test images.

The main idea behind **MobileNets** is to replace standard convolutions with depthwise separable convolutions in order to reduce calculations.
The network has 28 layers, the baseline network contains 3.23 million parameters. 
Attached is the MobileNet body architecture as it apppears in the papaer "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications":

![arch](/architecture.png)


MobileNet has two hyperparameters that trade off between size and accuracy of the network:
**width multiplier** shrinks the number of channels and thins down the netwrok at each layer, and
**resolution multiplier** reduces the image resolution. Originally, both parameteres are in the range (0, 1].

In this project, **depth multiplier** of 4 is used instead of resolution multiplier, as described in the paper "An Enhanced Hybrid MobileNet".
This solution has 0.85 million parameters, while the baseline model has 3.23 million parameters. 
I used 30% of dropout, SGD optimizer with momentum 0.9 and added augmentations to the training images. 
These improved the model proposed in the paper and achieved 75% of accuracy, which is slightly higher than the result achieved in the experiment.

