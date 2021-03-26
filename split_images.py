import mel_spectrograms as msg
from PIL import Image
from pathlib import Path


def split_image(file_path, name, x_start, x_end):
    img = Image.open(file_path)
    area = (x_start, 0, x_end, 217)
    cropped_img = img.crop(area)
    cropped_img.save(name)


def main():
    genres_folder_path = Path.cwd() / 'genres' / 'train'  # or './genres'
    genres_folder_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    jpg_paths = msg.get_all_paths(genres_folder_path, genres_folder_list, '.jpg')

    # apparently there are several sub lists in wav_paths, so this puts every path in list on the same level
    all_paths = []
    for i in jpg_paths:
        for path in i:
            all_paths.append(path)

    print('Paths have been collected.')

    # splitting images. This may take a while.
    print('Splitting in process.')
    for path in all_paths:
        for i in range(1, 11):
            width_start = (i-1)*33
            width_end = i*33
            split_image(path, str(path) + str(i) + str('.jpg'), width_start, width_end)


if __name__ == '__main__':
    # split_image(Path.cwd() / 'genres' / 'train' / 'blues' / 'blues.00000.wav_mel.jpg', 'split_test.jpg', 0, 33)
    main()
