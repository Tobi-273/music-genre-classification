# music-genre-classification
This project aims to classify a variety of musical genres in the given dataset. 30 second audio files are converted into spectrograms and analyzed using a convolutional neural network.

## Dataset
This project uses the GTZAN genre collection dataset (available under http://marsyas.info/downloads/datasets.html), which is part of marsyas.
Marsyas (Music Analysis, Retrieval and Synthesis for Audio Signals) is distributed under the GNU Public Licence (GPL) Version 2. A commercial license is also available for use in industrial projects and collaborations that do not wish to use the GPL license. Interested parties should contact George Tzanetakis (gtzan@cs.uvic.ca) for the specific details.

Please find our edited dataset here: https://drive.google.com/drive/folders/1e_YT-qif6Klq6wSaikc0LQG-VxYnZsCp?usp=sharing

## How To Use
1. Create Spectrograms: Download the GZTAN genre collection dataset from the link above, extract and move/copy the "genres" folder into the repository folder. Then run "mel_spectrograms.py". The spectrograms are in the .jpg format and will be generated in the same folders as the corresponding .wav files. The conversion process may take a while.
2. Not yet implemented

## Authors
Code by samerath, j-wicklein and Tobi-273.

## Contributing
Please note that this is a university project, so we can not accept any contributions until this is graded (March 2021). Afterwards, Pull Requests are welcome. Please elaborate on what you would like to change.
