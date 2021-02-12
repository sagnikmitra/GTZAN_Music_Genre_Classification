# GTZAN Music Genre Classification
Handling sound files in python, compute sound and audio features from them, and run machine learning algorithms on them and Classify the Audio Files according to their Genres.
# Objective of the Project:
### In this Project we will see how to handle sound files in python, compute sound and audio features from them, and run machine learning algorithms on them and Classify the Audio Files according to their Genres.
# Roadmap Followed:

* Loading and Decoding the Audio as a Time Series as a NumPy array with a default sampling rate(SR)
* Visualizing the Audio File (Decoded into an Arraylike Time Series Data) with Raw Wave plot or the amplitude envelope of audio waveform
* Plotting a Spectorgram to visually representing the signal loudness, of a signal over time at various frequencies present in a particular waveform. 
* Analyzing the Zero-Crossing Rate that can occur if successive samples have different algebraic signs. The rate at which zero crossings occur is a simple measure of the frequency content of a signal. 
* Plotting Chromagram to analyze music whose pitches can be meaningfully categorized and whose tuning approximates to the equal-tempered scale. 
* Preprocessing the data with SKLearn by encoding the label column with the function LabelEncoder() and splitting into test and train to obtain the Test Loss & Accuracy Score
# List of Libraries Used
**pandas :** is a library written for the Python programming language allowing data manipulation and analysis. In particular, it provides data structures and operations for manipulating numerical arrays and time series.

**numpy :** is an extension of the Python programming language, designed to manipulate multidimensional matrices or arrays as well as mathematical functions operating on these arrays.

**matplotlib :** Matplotlib is a library of the Python programming language for plotting and visualizing data in graphical form. It can be combined with the NumPy and SciPy python libraries for scientific computation.

**Scipy :** is a project aiming to unify and federate a set of Python libraries for scientific use. Scipy uses the arrays and matrices of the NumPy module.

**Pickle :** is a python module that allows you to save one or more variables in a file and retrieve their values later. Variables can be of any type.

**librosa :** It is a Python module to analyze audio signals in general but geared more towards music. It includes the nuts and bolts to build a MIR(Music information retrieval) system.

**IPython.display :** lets you play audio directly in your notebook.

**Librosa :**  is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.
# About the Dataset
Link to the Dataset:
https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification
### genres original - 
#### A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)

### images original - 
#### A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.

### CSV files - 
#### Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.
## Functionality of Librosa.load()

```
data , sr = librosa.load(audio_data)
```

loads and decodes the audio as a time series y, represented as a one-dimensional NumPy floating point array. The variable sr contains the sampling rate of y, that is, the number of samples per second of audio. By default, all audio is mixed to mono and resampled to 22050 Hz at load time. This behavior can be overridden by supplying additional arguments to librosa.load().

Audio will be automatically resampled to the given rate (default sr=22050).

To preserve the native sampling rate of the file, use sr=None.

We can change this behavior by resampling at 45.6KHz.
# Training Our Model

* Preparing a Four Layered Neural Network with the Four Consecutive Layers having 256, 128, 64, 10 Nodes respectively.
* Using Rectified Linear Unit as the Activation Function in the first three layers and SOFTMAX in the Output a.k.a. the Final Layer and Adam as the Optimizer of the Neural Network
* Train Test Spliting in a 67:33 Ratio and reshaped the Spectrogram’s Data into a One Dimentional Data and feeded in a densely packed Neural Network.
