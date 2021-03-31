README

# Machine Conditions Monitoring


## Table of contents
- [Introduction](#introduction)
- [Technologies](#technologies)
- [Setup](#setup)
- [Conclusion](#conclusion)



## 1. Introduction

Acme Corporation is a worldwide supplier of technological equipment. They are confronted by persistent downtimes, owing to unforeseen breakdown of machines within the production line. The problem is further exacerbated by absence of prior, timely maintenance, as a consequence, millions of US dollars in revenue are lost.   

In this project, we sought to build a robust predictive model, capable of forecasting potential breakdowns amongst different machine components based solely on input audio recordings. The audio files represent machines operating under normal and anomalous conditions. With the advent of such a model, it is expected that potential breakdowns in machine components can be forecasted, and thus intercepted before full scale damage halts operations of the manufacturing line.



## 2. Technology

This project was created with:

- Anaconda virtual environment version: 4.9.2
- anaconda-navigator version: 1.10.0
- Python version: 3.8.8
- Librosa library version 0.8.0
- Scikit-learn library version 0.24.1
- Jupyter notebook version: 6.2.0
- Numpy library version: 1.19.2
- Matplotlib library version: 3.3.4
- Pandas library version: 1.2.1
- Seaborn library version 0.11.0 


## 3. Setup

### i Downloading the data:

- Run ``download.py`` or manually download all files and unzip them in the ``data`` folder.
- Remove zip files, as they take up a lot of memory

### ii Feature exploration and visualization 

Using the Librosa library, diverse features of the audio files were extracted and thus visualized.
Visuals were crucial to generating deeper insights into delineations between normal and anomalous machine sounds. Graphics of extracted features can be accessed at: [Visuals](https://github.com/makyeme/Machine_conditions_monitoring/tree/DevelopmentMartin/Visuals). 

The code for audio visualization can be accessed here: [Code](https://github.com/makyeme/Machine_conditions_monitoring/blob/Development/simon_the_explorer.ipynb).
Below is an example depicting a graphical visualization of normal Vs abnormal sounds for a pump under diverse conditions of background noise:



![Optional Text](https://github.com/makyeme/Machine_conditions_monitoring/blob/DevelopmentMartin/Visuals/Raw_AudioWaves/AudioWave_pump.PNG)


Libraries used: Librosa, Matplotlib, Seaborn

### iii Feature Extraction and Generation

The lead up to model building necessitated extraction and generation of features.
The code used for extraction can be accessed at: [Code](https://github.com/makyeme/Machine_conditions_monitoring/blob/Development/simon_the_explorer.ipynb) 
Sample features are briefly elaborated below:

#### The Short-Term Fourier Transform(STFT)

- A very crucial aspect of time series signal processing. The STFT was used to  cut the audio waveform into short, overlapping equal length segments and take the Fourier transform of each segment individually to produce multiple power spectrograms, identifying resonant frequencies present in our audio file. 

#### Mel-Frequency Cepstral Coefficients(MFCCs)

- In brief, MFCC is a mathematical method which transforms the power spectrum of an audio signal to a small number of coefficients representing power of the audio signal in a frequency region (a region of pitch) taken relative to time., MFC coefficients give us an idea of the changing pitch of an audio signal.

#### The Chromagram 

- A chromagram is a representation of an audio signal w.r.t. time, mapping audio signal to a pitch class. Most often, we map to the 12 standard pitch classes (i.e. the musical scale CDEFGAB + 5 semitones gives us 12 pitch classes).

Graphics of a Chromagram STFT for the normal Vs abnormal sound of the pump are presented below:

![Optional Text](https://github.com/makyeme/Machine_conditions_monitoring/blob/DevelopmentMartin/Visuals/Audio_features/chroma_STFT_pump.PNG)



####  mel spectrogram 

- This is a spectrogram where the frequencies are converted to the mel scale.

#### The Root Mean Square Energy

- The square root of the mean of the square. RMS is a meaningful way of calculating the average of values over a period of time. With audio, the signal value (amplitude) is squared, averaged over a period of time, then the square root of the result is calculated. The result is a value, that when squared, is related (proportional) to the effective power of the signal. A plot of the RMSE for the pump is presented below:

![Optional Text](https://github.com/makyeme/Machine_conditions_monitoring/blob/DevelopmentMartin/Visuals/Audio_features/RMSE_pump.PNG)


#### Zero-Crossing Rate

- The Zero-Crossing Rate (ZCR) of an audio frame is the rate of sign-changes of the signal during the frame. In other words, it is the number of times the signal changes value, from positive to negative and vice versa, divided by the length of the frame
Libraries used: Librosa


### iv Data Preprocessing

Data preprocessing for machine learning  involved several operations as briefly explained below:

- Normal and abnormal sounds were one-hot encoded into binary classes (0, 1), these later constituted the target classes for the algorithm

- The data was separated into features (independent variables) and a binary target (dependent variable) 

- Features and binary target were converted to numpy arrays, a format largely accepted by the ML algorithms

- Data was systematically split along ID (to offset any bias) into random sets of: training, test and validation in the ratio: 

- Random seed was set at 42 to ensure consistent results across runs

Libraries used: Scikit-learn


### v Model training, Testing and Validation

Using the Scikit-learn library, several classification models were created, trained and tested on respective datasets. 
As an extra measure, a validation dataset was set aside for model evaluation and authentication after development through training and testing.

Model performance was evaluated via a series of matrices which include but not limited to:
- Accuracy,
- Classification report which contains precision and recall
- Confusion matrix 
- ROC curve

Based on performance metrices,  the two most suitable models for audio processing were:
- RandomForest
- SVC
- Logistic regression

Performance of select models was plotted on the ROC curve and is presented below: 


![Optional Text](https://github.com/makyeme/Machine_conditions_monitoring/blob/DevelopmentMartin/Visuals/ROC_curve/rocfinal.png)


Libraries: Scikit-learn


### vi Feature trimming and hyperparameter tunning

#### Feature trimming

Feature trimming/selection was done via the Scikit-learn library.
Feature importance was calculated, thus, basing on the results, highly weighted (important) features were retained for use in the final model.


#### Hyperparameter tunning



## 4. Conclusion

(closing remarks on developed model, Lessons learnt, takeaways, etc…?????)
- RandomForest was found to be the most robust classifier, with an accuracy of?? 
- The recall of the model could be improved 


### Limitations:


### Future directions/recommendations/Potential improvements:

- Make report on validity of pings(warnings) by model as feedback to improve model
