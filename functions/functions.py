import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import glob


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

def read_data(dirname='data_features/*.csv', scaling=True, cv=True, ID=False):
    csv_paths = glob.glob(dirname)
    df_list = []

    for path in csv_paths:
        df_list.append(pd.read_csv(path, index_col=0))
    df = pd.concat(df_list)
    df.dropna(inplace=True)
    if scaling:
        cols_scaling = df.drop('path', axis=1).columns
        scaler = StandardScaler()
        df[cols_scaling] = scaler.fit_transform(df[cols_scaling])
    df.loc[:, 'machine'] = df['path'].apply(lambda string: string.split('/')[2])
    df.loc[:, 'abnormal'] = df['path'].apply(lambda string: 1 if 'abnormal' in string else 0)
    df.loc[:, 'id'] = df['path'].apply(lambda string: string.split('/')[3][3:])
    dummies1 = pd.get_dummies(df['machine'], prefix='Machine')
    dummies2 = pd.get_dummies(df['id'], prefix='id')
    df = pd.concat([df, dummies1, dummies2], axis=1)
    if not id:

        df.drop(['machine', 'id'], axis=1, inplace=True)


    X = df.drop(['path', 'abnormal'], axis=1)
    y = df['abnormal'].values
    x_train_test, x_val, y_train_test, y_val = train_test_split(X, y, test_size=0.1,  random_state=42)
    if cv:
        return x_train_test, x_val, y_train_test, y_val
    x_train, x_test, y_train, y_test = train_test_split(x_train_test, y_train_test, test_size=0.15,  random_state=42)
    return x_train, x_test, x_val, y_train, y_test, y_val

read_data(dirname='data_features/*.csv', scaling=True)