# Google colab用です

from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD # 確率的勾配降下法
from keras.utils import np_utils

from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback

# network and training
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # number of outputs = number if digits
OPTIMIZER = SGD()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION

# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test  = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')

# normalize
X_train /= 255
X_test  /=255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary classmatrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test  = np_utils.to_categorical(y_test, NB_CLASSES)

# 10 outputs
# final stage is softmax
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.summary()


import os
from time import gmtime, strftime
from keras.callbacks import TensorBoard


def make_tensorboard(set_dir_name = ''):
  tictoc = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())
  directory_name = tictoc
  log_dir = set_dir_name + '_' + directory_name
  os.mkdir(log_dir)
  tensorboard = TensorBoard(log_dir=log_dir)
  return tensorboard


tbc=TensorBoardColab()

model.compile(loss = 'categorical_crossentropy',
                          optimizer = OPTIMIZER,
                          metrics =  ['accuracy'] )


model.fit(X_train,
                 Y_train,
                 batch_size = BATCH_SIZE,
                 epochs = NB_EPOCH,
                 callbacks = [TensorBoardColabCallback(tbc)],
                 verbose = VERBOSE,
                 validation_split = VALIDATION_SPLIT
                )

score = model.evaluate(X_test, Y_test, verbose = VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])


# calculate predictions
predictions = model.predict(X_test)

