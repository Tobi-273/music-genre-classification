from tensorflow import keras
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing import image as img

def classifier():
    model_path = Path.cwd() / 'yt_classifier' / 'saved_model'
    model = keras.models.load_model(model_path)

    image_path = Path.cwd() / 'yt_classifier' / 'spectrogram.jpg'
    image = img.load_img(image_path, target_size=(217, 334), color_mode='rgb')
    image = np.asarray(image)
    image = image[:,:,:1]
    image = np.expand_dims(image, axis=0)/255

    result = model.predict(image)

    max_index_row = np.argmax(result, axis=1)
    genre_number = max_index_row[0]
    genre_number = genre_number.item()

    genres = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

    print(genres[genre_number])

classifier()