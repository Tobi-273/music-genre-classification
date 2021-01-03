''' [first incomplete draft of] a simple feed-forward (non-convolutional) neural network with our dataset'''

import tensorflow as tf
import os
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

blues_dir = os.path.join('./genres_mini_training/blues')
classical_dir = os.path.join('./genres_mini_training/classical')

print('total blues spectrograms:', len(os.listdir(blues_dir)))
print('total classical spectrograms:', len(os.listdir(classical_dir)))

complete_dir = os.path.join('./genres_mini_training')
print('total spectrograms: ', len(os.listdir(complete_dir)))

'''
TRAINING_DIR = "./genres_mini_training"
training_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126)
'''