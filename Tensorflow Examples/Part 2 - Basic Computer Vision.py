# from https://colab.research.google.com/github/lmoroney/
# mlday-tokyo/blob/master/Lab2-Computer-Vision.ipynb#scrollTo=WzlqsEzX9s5P

import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

# loading the dataset
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
print(f'Der Dateityp von training_images ist {type(training_images)}')
print(f'Der Dateityp von training labels ist {type(training_labels)}')
print(training_labels)

# normalizing the training data
training_images = training_images / 255.0
test_images = test_images / 255.0

# design the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# build the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(training_images, training_labels, epochs=5)

# evaluate the model
model.evaluate(test_images, test_labels)


# Exercises can be found in the original jupyter notebook (see at the top)

# Exercise 8 (Code with Callbacks):
'''
Earlier when you trained for extra epochs you had an issue where your loss might change. It might have taken a bit of 
time for you to wait for the training to do that, and you might have thought 'wouldn't it be nice if I could stop the 
training when I reach a desired value?' -- i.e. 95% accuracy might be enough for you, and if you reach that after 
3 epochs, why sit around waiting for it to finish a lot more epochs....So how would you fix that? Like any other 
program...you have callbacks! Let's see them in action...

import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
'''
