from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras import utils
from keras import layers
import keras
import numpy as np
import os

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/255
x_test = x_test/255
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)


# create model
input_tensor = layers.Input(shape = x_train.shape[1:])
alpha = 0.25
depth = 4
model = MobileNet(input_shape=None, alpha=alpha, depth_multiplier=depth, dropout=0.1,
                  include_top=True, weights=None, input_tensor=input_tensor, pooling=None, classes=10)

# set meta params
batch_size = 128
nb_classes = 10
nb_epoch   = 2
nb_data    = 32*32

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=0.0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        vertical_flip=False,
        horizontal_flip=True)
datagen.fit(x_train)


# Prepare model model saving directory.
model_type = 'MobileNetV1_'
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

csv_name = 'cifar10_%s_model.csv' % model_type
csv_filepath = os.path.join(save_dir, csv_name)


# set learning rate
learning_rates=[]
for i in range(5):
    learning_rates.append(2e-2)
for i in range(50-5):
    learning_rates.append(1e-2)
for i in range(100-50):
    learning_rates.append(8e-3)
for i in range(150-100):
    learning_rates.append(4e-3)
for i in range(200-150):
    learning_rates.append(2e-3)
for i in range(300-200):
    learning_rates.append(1e-3)

# create callbacks to freeze best models and save training history
callbacks = []
callbacks.append(ModelCheckpoint(filepath, save_best_only=True))
callbacks.append(CSVLogger(csv_filepath))
callbacks.append(LearningRateScheduler(lambda epoch: float(learning_rates[epoch])))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False), metrics=['accuracy'])


# create fit function
history = model.fit_generator(
              datagen.flow(x_train, y_train, batch_size=128),
              steps_per_epoch=len(x_train) / 128,
              epochs=300,
              verbose=1,
              callbacks=callbacks,
              validation_data=(x_test, y_test))

# model.summary()
