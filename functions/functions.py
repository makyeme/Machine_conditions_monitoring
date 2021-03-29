import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import entropy



def abs_fft(wave: np.array):
    return np.abs(np.fft.fft(wave))

def simple_stats(x, key, feature=True, entropy=True, extra_info={}):
    data = {}

    if key == 'fft':
        for hz_range in extra_info['frequency_ranges']:
            n = hz_range[1] - hz_range[0]
            data[f'{key}_{hz_range}_sqrt_mean_square'] = np.sqrt(np.mean(np.square(
                x[hz_range[0]:hz_range[1]]))) / n
            data[f'{key}_{hz_range}_mean'] = np.mean(
                x[hz_range[0]:hz_range[1]]) / n


    data[key + f'{key}_mean'] = np.mean(x)
    data[f'{key}_median'] = np.median(x)
    data[f'{key}_std'] = np.std(x)
    print(type(kurtosis(x)))
    data[f'{key}_kurtosis'] = kurtosis(x)
    data[f'{key}_skew'] = skew(x)
    data[f'{key}_min'] = np.min(x)
    data[f'{key}_max'] = np.max(x)
    data[f'{key}_amp'] = data[f'{key}_max'] - data[f'{key}_min']
    data[f'{key}_SNR'] = data[f'{key}_median'] / data[f'{key}_std']
    if feature:
        data[f'{key}_entropy'] = entropy(x, axis=1)[0]
        data[f'{key}_kurtosis'] = kurtosis(x, axis=1)[0]
        data[f'{key}_skew'] = skew(x, axis=1)[0]
    else:
        print('x', x)
        print('data', data)
        print('key', key)
        data[f'{key}_entropy'] = entropy(x, axis=1)
        data[f'{key}_kurtosis'] = kurtosis(x, axis=1)
        data[f'{key}_skew'] = skew(x, axis=1)
    ## Percentiles
    for number, percent in zip(np.percentile(x, extra_info['percentiles']), extra_info['percentiles']):
        data[f'{key}_percentile_{percent}'] = number
    return data
