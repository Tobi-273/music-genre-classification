from pathlib import Path
from pydub import AudioSegment
import glob

song_dir = Path.cwd() / 'genres_mini_training(.wav)' / 'blues'

# function to split songs into snippets and export file
def split_songs(song, from_sec, to_sec, split_filename):
  t1 = from_sec
  t2 = to_sec
  newAudio = AudioSegment.from_wav(song)
  newAudio = newAudio[t1:t2] # milliseconds
  newAudio.export(split_filename, format="wav") # Exports to a wav file in the current path.
  return newAudio.export(split_filename, format="wav")


# iterate over folder directory of songs to split all of them and safe new snippets
songs = glob.glob('genres/blues' + '/*.wav')
i = 0
for s in songs:
  print(s)
  for t in range(0,30, 3): # songs are 30 seconds long, split into snippets of 3 s
    t=t*1000
    split_songs(s, t, t+3000, "Splitsong{number}_{time}.wav".format(number = i, time=t))
    # print(split_songs(s, t, t+3000, "Song{number}_{time}.wav".format(number = i, time=t)))
  i= i+1
