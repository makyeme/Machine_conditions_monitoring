import pandas as pd
import librosa
import librosa.display
import glob
from functions.functions import abs_fft
from functions.functions import simple_stats
from time import time


import os

SR = 16000
PERCENTILES = [10*i for i in range(1, 10)]
FREQUENCY_RANGES = [(0, 5),
                    (5, 20),
                    (20, 50),
                    (50, 200),
                    (200, 500),
                    (500, 2000),
                    (2000, 5000),
                    (5000, 20000),
                    (20000, 50000),
                    (50000, 160000)]


class Features:
    """
    Class that will handle all feature extraction, cleaning, processing and augmentation from the raw .wav files.
    """

    def __init__(self, feature_functions=None, sr=16000):
        print('lolo')
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

    def extract_features(self, path):
        wave, sr = librosa.load(path, sr=self.sampling_rate)
        extra_info = {'frequency_ranges': FREQUENCY_RANGES, 'percentiles': PERCENTILES}
        data = {}
        # Some standard statistics you can take from an array, a lot of duplicate code that can be removed, made better
        # percentiles also contain the median (and could contain the min and max as well but not currently)

        # Statistics on the wave itself
        data.update(simple_stats(x=wave,
                                 key='wave',
                                 extra_info=extra_info))

        for key, function in self.feature_functions.items():
            if key == 'fft':
                transformed_wave = function(wave)
                data.update(simple_stats(transformed_wave, key=key, extra_info=extra_info))
                continue
            try:
                transformed_wave = function(y=wave, sr=self.sampling_rate)
            except TypeError:
                transformed_wave = function(y=wave)

            data.update(simple_stats(transformed_wave, key=key, extra_info=extra_info))

        data['path'] = path
        return data

    def extract_features_from_machine(self, machine, save=True):
        list_paths_machine = glob.glob(f'data/*/{machine}/*/*/*.wav')
        length = len(list_paths_machine)
        print(f'files: {length}')
        t0 = time()
        data_list = []
        for i, path in enumerate(list_paths_machine):
            if i % 200 == 1:
                p_done = i / length * 100
                print(f'****************** {machine} ******************')
                print(f'****************** {machine} ******************')
                print(f'****************** {machine} ******************')
                print("{:.2f}%".format(p_done))
                t1 = time()
                td = t1 - t0
                time_left = td / i * (length - i)
                print(f'Estimated time left: {int(time_left / 60 * 100) / 100} minutes')
            data_list.append(self.extract_features(path))
        df = pd.DataFrame(data_list)
        if save:
            if not os.path.exists('data_features'):
                os.makedirs('data_features')
            df.to_csv(f'data_features/features_{machine}.csv')
        return df

    
