# Google colab用です

import os
import keras
from keras.models import Sequential # 層を積み重ねたモデル
from keras.layers.convolutional import Conv2D # 畳み込み層
from keras.layers.convolutional import MaxPooling2D # 最大プーリング層
from keras.layers.core import Activation # 出力に活性化関数を適用する
from keras.layers.core import Flatten, Dropout # 全結合、ドロップアウト層
from keras.layers.core import Dense # 通常の全結合ニューラルネットワークレイヤー
from keras.datasets import mnist # データの読み込み
from keras.optimizers import Adam # 最適化の手法としてAdamを適用
from keras.callbacks import TensorBoard # TensorBoard
from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback # ColabでTensorBoardを使えるようにする

# LeNetのネットワークの定義
def lenet(input_shape, num_classes):
  model = Sequential()
  # extract image features by convolution and max pooling layers
  model.add(Conv2D(20, 
                      kernel_size=5,
                      padding="same",
                      input_shape=input_shape,
                      activation="relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(50,
                      kernel_size=5,
                      padding="same",
                      activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # classify the class by fully-connected layers
  model.add(Flatten())
  model.add(Dense(500, activation="relu"))
  model.add(Dense(num_classes))
  model.add(Activation("softmax"))
  return model


# 学習用データセット
class MNISTDataset():
  def __init__(self):
    self.image_shape = (28, 28, 1) # image is 28x28x1(gray scale)
    self.num_classes = 10
    
  def get_batch(self):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
    y_train, y_test = [self.preprocess(d, label_data = True) for d in [y_train, y_test]]
    return x_train, y_train, x_test, y_test
  
  def preprocess(self, data, label_data = False):
    if label_data:
      # convert class vectors yo binary class matrices
      data = keras.utils.to_categorical(data, self.num_classes)
    else:
      data = data.astype("float32")
      data /= 255 # convert the value to 0-1 scale
      shape = (data.shape[0], ) + self.image_shape # add dataset length to top
      data = data.reshape(shape)
    return data    

tbc=TensorBoardColab()


# ネットワークを学習させるTrainer    
class Trainer():
  def __init__(self, model, loss, optimizer, logdir="logdir"):
        self.target = model
        self.target.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        self.verbose = 1
        self.log_dir = os.path.join(os.path.dirname(__file__), logdir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
  def train(self, x_train, y_train, batch_size, epochs, validation_split):
        if os.path.exists(self.log_dir):
          import shutil
          shutil.rmtree(self.log_dir) # remove previous execution
      
        os.mkdir(self.log_dir)
    
        self.target.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                         validation_split=validation_split,
                         callbacks=[TensorBoardColabCallback(tbc)],
                        verbose=self.verbose)


dataset = MNISTDataset()

# make model
model = lenet(dataset.image_shape, dataset.num_classes)

# train the model
x_train, y_train, x_test, y_test = dataset.get_batch()
trainer = Trainer(model, loss="categorical_crossentropy",
                              optimizer=Adam(),  logdir="logdir32")
trainer.train(x_train, y_train, batch_size=128, epochs=12, validation_split=0.2)

# show result
score = model.evaluate(x_train, y_train, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


