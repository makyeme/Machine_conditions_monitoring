import pandas as pd
import numpy as np
import librosa
import IPython.display as ipd #palying audio
import matplotlib.pyplot as plt
import librosa.display
import csv
import glob
from functions.functions import abs_fft
from functions.functions import simple_stats


import os

SR = 16000
PERCENTILES = [10*i for i in range(1, 10)]
FREQUENCY_RANGES = [(0, 5), (5, 20), (20, 50), (50, 200), (200, 500), (500, 2000), (2000, 5000), (5000, 20000),
                             (20000, 50000), (50000, 160000)]


class Features:
    """
    Class that will handle all feature extraction, cleaning, processing and augmentation from the raw .wav files.
    """

    def __init__(self, feature_functions=None, sr=16000):
        self.sampling_rate = sr
        if isinstance(feature_functions, type(None)):
            self.feature_functions = {
                'chroma_stft': librosa.feature.chroma_stft,
                'rms': librosa.feature.rms,
                'spec_cent': librosa.feature.spectral_centroid,
                'spec_bw': librosa.feature.spectral_bandwidth,
                'rolloff': librosa.feature.spectral_rolloff,
                'zcr': librosa.feature.zero_crossing_rate,
                'mfcc': librosa.feature.mfcc,
                'fft': abs_fft,
            }

    def extract_feature(self, path):
        wave, sr = librosa.load(path, sr=self.sampling_rate)

        data = {}
        percentiles = [10 * i for i in range(1, 10)]
        frequency_ranges = [(0, 5), (5, 20), (20, 50), (50, 200), (200, 500), (500, 2000), (2000, 5000), (5000, 8000)]

        # Some standard statistics you can take from an array, a lot of duplicate code that can be removed, made better
        # percentiles also contain the median (and could contain the min and max as well but not currently)

        # Statistics on the wave itself
        data.update(simple_stats(x=wave,
                                 key='wave',
                                 feature=False,
                                 entropy=False,
                                 extra_info=FREQUENCY_RANGES))

        for key, function in self.feature_functions.items():
            if key == 'fft':
                transformed_wave = function(wave)
                data.update(simple_stats(transformed_wave, key='key', feature=False, entropy=False))
                continue
            try:
                transformed_wave = function(y=wave, sr=self.sampling_rate)
            except TypeError:
                transformed_wave = function(y=wave)


                    # Ugly if statements, could be done better
            if transformed_wave.ndim < 2:
                data.update(simple_stats(transformed_wave, key='key', feature=True))

            else:
                mtw = np.mean(transformed_wave, axis=1)
                for i, row in enumerate(mtw):
                    data[f'{key}_mean_{i}'] = row

        data['path'] = path
        return data

