from __future__ import unicode_literals
import youtube_dl
from pydub import AudioSegment
import os
from pathlib import Path
from mel_spectrograms import create_mel

# download mp3 audio from youtube link
print("Insert the YouTube link")
link = input ("")

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '320',
    }],
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([link])


# getting path of downloaded mp3 file
iterable = Path.cwd().iterdir()
for j in iterable:
    if j.suffix == '.mp3':
        path = j

# selecting 30 second segment
startMin = 0
startSec = 00

endMin = 0
endSec = 30

# Time to miliseconds
startTime = startMin*60*1000+startSec*1000
endTime = endMin*60*1000+endSec*1000

# Opening file and extracting segment
song = AudioSegment.from_mp3(path)
extract = song[startTime:endTime]

# Saving
extract.export(Path.cwd() / 'extraction' / 'extraction.wav', format="wav")


# getting paths of mp3(s)
new_paths = []
iterable = Path.cwd().iterdir()
for j in iterable:
    if j.suffix == '.mp3':
        new_paths.append(j)

# deleting them
for i in new_paths:
    os.remove(i)


# create spectrogram
create_mel(Path.cwd() / 'extraction' / 'extraction.wav', 'spectrogram.jpg')
