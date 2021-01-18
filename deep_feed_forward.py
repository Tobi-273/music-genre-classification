# batch size, epochs, loss, metrics, optimizer, model

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')

train_dir = Path.cwd() / '/content/drive/MyDrive/AAA Private Ablage/Dateien/Studium/Leuphana/Python/genres/train'
test_dir = Path.cwd() / '/content/drive/MyDrive/AAA Private Ablage/Dateien/Studium/Leuphana/Python/genres/test'

IMG_HEIGHT = 217
IMG_WIDTH = 334
epochs = 50
batch_size = 1

img_data_gen = ImageDataGenerator(rescale=1./255)

train_data_gen = img_data_gen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

validation_data_gen = img_data_gen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical',
    subset='validation')  # set as validation data

test_data_gen = img_data_gen.flow_from_directory(
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')


# design the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(2048, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# build the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_data_gen, epochs=epochs, batch_size=1, validation_data=validation_data_gen)
model.evaluate(test_data_gen)
model.summary()
model.save("saved_model")

# plot training and validation loss
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs_loss = range(1, (epochs+1))
plt.plot(epochs_loss, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot training and validation accuracy
loss_train = history.history['accuracy']
loss_val = history.history['val_acc']
epochs_accuracy = range(1, (epochs+1))
plt.plot(epochs_accuracy, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
