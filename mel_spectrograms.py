import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# noinspection SpellCheckingInspection
def create_mel(wav_file_path, mel_file_name):
    """Takes a path to a .wav file and creates a mel spectrogram in the cwd (current working directory)"""

    y, sr = librosa.load(wav_file_path)

    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    # In another example for using this package (but not related to this task: (y=y, sr=sr, n_mels=128, fmax=8000)

    mel_spect = librosa.power_to_db(s, ref=np.max)

    librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
    # The other example sets only fmax (to the same value), function of other kwargs unclear

    plt.savefig(mel_file_name)


def get_file_paths_in_folder(given_folder_path, suffix='.wav'):
    new_wav_paths = []
    iterable = given_folder_path.iterdir()
    for j in iterable:
        if j.suffix == suffix:
            new_wav_paths.append(j)
    return new_wav_paths


def get_sub_path(path, sub_folder_name):
    new_path = path / sub_folder_name
    return new_path


def get_wav_paths(base_path, folders):
    folder_paths = []
    for j in folders:
        sub_path = get_sub_path(base_path, j)
        folder_paths.append(get_file_paths_in_folder(sub_path))
    return folder_paths


def mel_file_name_creator(wav_file_name):
    mel_file_name = wav_file_name + '_mel.jpg'
    return mel_file_name


genres_folder_path = Path.cwd() / 'genres' #'./genres' # or Path.cwd() / 'genres'
genres_folder_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
wav_paths = get_wav_paths(genres_folder_path, genres_folder_list)

all_paths = []
for i in wav_paths:
    for path in i:
        all_paths.append(path)


for path in all_paths:
    mel_name = mel_file_name_creator(str(path))
    create_mel(path, mel_name)