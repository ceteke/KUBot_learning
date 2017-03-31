import numpy as np
from copy import deepcopy
from math import sqrt
from sample import Sample


def scale_sample(sample):
    new_sample = deepcopy(sample)
    new_sample.X /= np.max(np.abs(sample.X), axis=0)
    new_sample.y /= np.max(np.abs(sample.y), axis=0)
    return new_sample


def get_average(arr):
    total = 0.0
    for a in arr:
        total += a
    return total/len(arr)


def get_std(arr, avg):
    total = 0.0
    for a in arr:
        total += (a-avg)**2
    return sqrt(total/len(arr))


def z_score_scale(arr):
    avg = get_average(arr)
    std = get_std(arr, avg)
    for i in range(len(arr)):
        arr[i] = (arr[i] - avg)/std


def z_score_sample_scale(sample):
    X_histogram = list(sample.X[24:69])
    y_histogram = list(sample.y[24:69])

    z_score_scale(X_histogram)
    z_score_scale(y_histogram)

    X = list(sample.X[0:24])
    y = list(sample.y[0:24])

    X += X_histogram
    y += y_histogram
    return Sample(X, y, sample.obj)


def zero_one_scaler(sample):
    X_histogram = list(sample.X[24:69])
    y_histogram = list(sample.y[24:69])

    Xmax = max(X_histogram)
    Xmin = min(X_histogram)

    ymax = max(X_histogram)
    ymin = min(X_histogram)

    for i in range(len(X_histogram)):
        X_histogram[i] = (X_histogram[i] - Xmin) / (Xmax - Xmin)
        y_histogram[i] = (y_histogram[i] - ymin) / (ymax - ymin)

    X = list(sample.X[0:24])
    y = list(sample.y[0:24])

    Xmax = max(X)
    Xmin = min(X)

    ymax = max(y)
    ymin = min(y)

    for i in range(len(X)):
        X[i] = (X[i] - Xmin) / (Xmax - Xmin)
        y[i] = (y[i] - ymin) / (ymax - ymin)

    X += X_histogram
    y += y_histogram

    return Sample(np.array(X), np.array(y), sample.obj)
