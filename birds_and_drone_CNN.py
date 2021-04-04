import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np



model=Sequential()
model = Sequential()
model.add(Conv2D( filters =16, kernel_size= (2,2), activation= 'relu', input_shape=(512,512,3)))
model.add( MaxPool2D( 2 , 2 ) ) 
model.add( Conv2D ( 32,(3,3), activation='relu' ) )
model.add( MaxPool2D( 2 , 2 ) )
model.add( Conv2D ( 64,(3,3), activation='relu' ) )
model.add( MaxPool2D( 2 , 2 ) )

model.add( Flatten()  ) 
model.add( Dropout( 0.4 ) )
model.add(Dense( units = 512, activation= 'relu'))
model.add(Dense( units = 128, activation= 'relu'))
model.add(Dense(units = 1, activation='sigmoid'))


model.summary()

from keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = '/content/drive/MyDrive/Colab Notebooks/drone_bird/training'
train_datagen = ImageDataGenerator(rescale = 1/255.0,
                                   rotation_range=40, 
                                   width_shift_range=0.2, 
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2, 
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   fill_mode = 'nearest')


train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size = 10, class_mode = 'binary', target_size=(512,512))
VALIDATION_DIR = '/content/drive/MyDrive/Colab Notebooks/drone_bird/testing'
validation_datagen = ImageDataGenerator(rescale = 1/255.0)


validation_generator =validation_datagen.flow_from_directory(VALIDATION_DIR, batch_size = 10, class_mode = 'binary', target_size=(512,512))


model.compile(loss='binary_crossentropy',optimizer = RMSprop(lr=0.001), metrics=['accuracy'])

history = model.fit(train_generator,
                              epochs=75,
                              verbose=1,
                              validation_data = validation_generator)


def learning_curve(history, epoch):
  # 정확도 차트
  plt.figure(figsize = (10, 5))

  epoch_range = np.arange(1, epoch + 1)

  plt.subplot(1, 2, 1)

  # history는 fit 결과값을 저장하는 변수
  plt.plot( epoch_range, history.history["accuracy"])
  plt.plot( epoch_range, history.history["val_accuracy"])
  plt.title("Model Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accurach")
  plt.legend( ["Train", "Val"] )


  # loss 차트
  plt.figure(figsize = (10, 5))

  plt.subplot(1, 2, 2)

  plt.plot( epoch_range, history.history["loss"])
  plt.plot( epoch_range, history.history["val_loss"])
  plt.title("Model Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend( ["Train", "Val"] )
  
  plt.show()

learning_curve(history, 75)
