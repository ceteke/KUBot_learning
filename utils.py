import numpy as np


def normalize_array(arr, r_max=1, r_min=0):
    arr_std = (arr - arr.min()) / (arr.max() - arr.min())
    arr_scaled = arr_std * (r_max - r_min) + r_min
    return arr_scaled


def scale_data_point(features):
    # hist3 = features[54:69]
    # hist2 = features[39:54]
    hist1 = features[24:69]
    others = features[0:24]

    # hist2 = normalize_array(hist2)
    # hist3 = normalize_array(hist3)
    others = normalize_array(others)
    hist1 = normalize_array(hist1)

    return np.concatenate((others, hist1))
