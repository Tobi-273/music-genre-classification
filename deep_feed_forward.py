# batch size, epochs, loss, metrics, optimizer, model

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path


train_dir = Path.cwd() / 'genres_mini_training_gray'
test_dir = Path.cwd() / 'genres_mini_test_gray'

IMG_HEIGHT = 217
IMG_WIDTH = 334

train_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

test_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')


# design the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# build the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Version 1
model.fit(train_data_gen, epochs=5)
model.evaluate(test_data_gen)
model.summary()
model.save("saved_model")

'''
# Version 2
history = model.fit(train_data_gen, epochs=25, steps_per_epoch=20, validation_data = test_data_gen, verbose = 1, 
validation_steps=3)

model.save("rps.h5")


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
    # predicting images
    path = fn
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(fn)
    print(classes)
'''