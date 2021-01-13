# preprocessing the data and labeling it

import numpy as np
import mel_spectrograms as msg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

np.set_printoptions(threshold=np.inf)

genres_folder_path = Path.cwd() / 'genres_mini_training'  # or './genres'
genres_folder_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
jpg_paths = msg.get_all_paths(genres_folder_path, genres_folder_list, '.jpg')
#data =

#def get_data():
img_path = jpg_paths[0][0]

img = mpimg.imread(img_path)
print(img)
print(img.shape)

plt.imshow(img)
plt.show()


'''
genres_folder_path = Path.cwd() / 'genres_mini_training (.wav)'  # or './genres'
genres_folder_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
wav_paths = get_all_paths(genres_folder_path, genres_folder_list, '.wav')

# apparently there are several sub lists in wav_paths, so this puts every path in list on the same level
all_paths = []
for i in wav_paths:
    for path in i:
        all_paths.append(path)

print('Paths have been collected.')

# generating the spectrograms as .jpg files in the folders corresponding to the .wav files. This may take a while.
print('Spectrogram generation in process.')
for path in all_paths:
    mel_name = mel_file_name_creator(str(path))
    create_mel(path, mel_name)


if __name__ == '__main__':
    main()
'''
