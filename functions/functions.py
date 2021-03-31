import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import entropy


def abs_fft(wave: np.array):
    return np.abs(np.fft.fft(wave))


def simple_stats(x: np.array, key: str, extra_info: dict) -> dict:
    data = {}

    if key == 'fft':
        for hz_range in extra_info['frequency_ranges']:
            n = hz_range[1] - hz_range[0]
            data[f'{key}_{hz_range}_sqrt_mean_square'] = np.sqrt(np.mean(np.square(
                x[hz_range[0]:hz_range[1]]))) / n
            data[f'{key}_{hz_range}_mean'] = np.mean(
                x[hz_range[0]:hz_range[1]]) / n

    data[f'{key}_mean'] = np.mean(x)
    data[f'{key}_median'] = np.median(x)
    data[f'{key}_std'] = np.std(x)
    data[f'{key}_min'] = np.min(x)
    data[f'{key}_max'] = np.max(x)
    data[f'{key}_amp'] = data[f'{key}_max'] - data[f'{key}_min']
    data[f'{key}_SNR'] = data[f'{key}_mean'] / data[f'{key}_std']
    if key == 'fft':
        data[f'{key}_entropy'] = entropy(x)
        data[f'{key}_kurtosis'] = kurtosis(x)
        data[f'{key}_skew'] = skew(x)
        return data
    if x.ndim == 2:
        if key != 'mfcc':
            data[f'{key}_entropy'] = entropy(x, axis=1).mean()
        data[f'{key}_kurtosis'] = kurtosis(x, axis=1).mean()
        data[f'{key}_skew'] = skew(x, axis=1).mean()
        mean_x = np.mean(x, axis=1)
        std_x = np.std(x, axis=1)
        for i in range(len(mean_x)):
            data[f'{key}_mean_{i}'] = mean_x[i]
            data[f'{key}_std_{i}'] = std_x[i]

    else:
        data[f'{key}_kurtosis'] = kurtosis(x)
        data[f'{key}_skew'] = skew(x)
        for number, percent in zip(np.percentile(x, extra_info['percentiles']), extra_info['percentiles']):
            data[f'{key}_percentile_{percent}'] = number
    return data
