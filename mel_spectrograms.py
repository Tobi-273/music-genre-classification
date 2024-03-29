"""Code for generating spectrograms"""

import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# noinspection SpellCheckingInspection
def create_mel(wav_file_path, mel_file_name):
    """Creates a mel spectrogram in the cwd (current working directory).
    Takes a path to a .wav file (datatype unclear) and a new name for the spectrogram as a string."""

    y, sr = librosa.load(wav_file_path)

    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    # In another example for using this package (but not related to this task: (y=y, sr=sr, n_mels=128, fmax=8000)

    mel_spect = librosa.power_to_db(s, ref=np.max)
    
    ax = plt.axes()
    ax.set_axis_off()

    librosa.display.specshow(mel_spect, fmax=8000, cmap='gray_r')
    
    plt.savefig(mel_file_name, bbox_inches='tight', transparent=True, pad_inches=0.0)
    # not sure what the inches mean, but it was just in an example and works


def get_file_paths_in_folder(given_folder_path, suffix=None):
    """Looks for all the folder paths with a certain suffix and returns those"""
    new_wav_paths = []
    iterable = given_folder_path.iterdir()
    for j in iterable:
        if j.suffix == suffix:
            new_wav_paths.append(j)
    return new_wav_paths


def get_all_paths(base_path, folders, suffix):
    """Goes through a list of given folder names and returns all paths with a given suffix"""
    folder_paths = []
    for j in folders:
        sub_path = base_path / j
        folder_paths.append(get_file_paths_in_folder(sub_path, suffix=suffix))
    return folder_paths


def mel_file_name_creator(wav_file_name):
    """Creates a new file name for the spectrogram"""
    mel_file_name = wav_file_name + '_mel.jpg'
    return mel_file_name


def main():
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
