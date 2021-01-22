# batch size, epochs, loss, metrics, optimizer, model, steps_per_epoch, early stopping, dropout, regularization

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from google.colab import drive

# for comparability of the different models
# import numpy as np
# np.random.seed(5)

drive.mount('/content/drive')

train_dir = Path.cwd() / '/content/drive/MyDrive/AAA Private Ablage/Dateien/Studium/Leuphana/Python/genres/train'  # Tobi
test_dir = Path.cwd() / '/content/drive/MyDrive/AAA Private Ablage/Dateien/Studium/Leuphana/Python/genres/test'  # Tobi

# train_dir =Path.cwd() / '/content/drive/MyDrive/UNI/Machine Learning/genres/train' #jana
# test_dir =Path.cwd() / '/content/drive/MyDrive/UNI/Machine Learning/genres/test' #jana

# train_dir = Path.cwd() / '/content/drive/MyDrive/Dies und Das/genres/train' #Sandra
# test_dir = Path.cwd() / '/content/drive/MyDrive/Dies und Das/genres/test' #Sandra

IMG_HEIGHT = 217
IMG_WIDTH = 334
epochs = 50
batch_size = 64

training_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.15)

train_data_gen = training_generator.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    seed=5,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_data_gen = training_generator.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    seed=5,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

test_data_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    seed=5,
    batch_size=batch_size,
    class_mode='categorical')

# design the model
model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
     tf.keras.layers.MaxPooling2D((2, 2)),

     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(2048, activation=tf.nn.relu),
     tf.keras.layers.Dense(1024, activation=tf.nn.relu),
     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# build the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data_gen,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=validation_data_gen,
    steps_per_epoch=train_data_gen.samples // batch_size,
    validation_steps=validation_data_gen.samples // batch_size,

)

model.evaluate(test_data_gen)
model.summary()
model.save("saved_model")

# plot training and validation loss
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs_loss = range(1, (epochs + 1))
plt.plot(epochs_loss, loss_train, 'g', label='Training loss')
plt.plot(epochs_loss, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot training and validation accuracy
loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs_accuracy = range(1, (epochs + 1))
plt.plot(epochs_accuracy, loss_train, 'g', label='Training accuracy')
plt.plot(epochs_accuracy, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
