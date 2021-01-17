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

sample_training_images, _ = next(train_data_gen)


# design the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# build the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_data_gen, epochs=5)

# evaluate the model
model.evaluate(test_data_gen)
model.summary()
model.save("saved_model")
'''
blues_dir = os.path.join('./genres_mini_training/blues')
classical_dir = os.path.join('./genres_mini_training/classical')

print('total blues spectrograms:', len(os.listdir(blues_dir)))
print('total classical spectrograms:', len(os.listdir(classical_dir)))

complete_dir = os.path.join('./genres_mini_training')
print('total spectrograms: ', len(os.listdir(complete_dir)))
'''
'''
TRAINING_DIR = "./genres_mini_training"
training_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=126)
'''