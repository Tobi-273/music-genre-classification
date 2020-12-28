import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_mel(wav_file_path, mel_file_name):
    """Takes a path to a .wav file and creates a mel spectrogram in the cwd (current working directory)"""

    y, sr = librosa.load(wav_file_path)

    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    # In another example for using this package (but not related to this task: (y=y, sr=sr, n_mels=128, fmax=8000)

    mel_spect = librosa.power_to_db(s, ref=np.max)

    librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
    # The other example sets only fmax (to the same value), function of other kwargs unclear

    plt.savefig(mel_file_name)


def get_file_paths_in_folder(folder_path, suffix='.wav'):
    new_wav_paths = []
    iterable = folder_path.iterdir()
    for i in iterable:
        if i.suffix == suffix:
            new_wav_paths.append(i)
    return new_wav_paths


def get_subpath(path, subfolder_name):
    subpath = path / subfolder_name
    return subpath


def get_wav_paths(basepath, folders):
    all_paths = []
    for i in folder_list:
        subpath = get_subpath(basepath, i)
        wav_paths.append(get_file_paths_in_folder(subpath))
    return wav_paths


def mel_file_name_creator(wav_file_name):
    return mel_file_name


folderpath = './genres' # or Path.cwd() / 'genres'
folder_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
wav_paths = get_wav_paths(folderpath, folder_list)

for i in wav_paths:
    mel_name = mel_file_name_creator(i)
    create_mel(i, mel_name)