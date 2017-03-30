import numpy as np
from copy import deepcopy

def scale_sample(sample):
    new_sample = deepcopy(sample)
    new_sample.X /= np.max(np.abs(sample.X),axis=0)
    new_sample.y /= np.max(np.abs(sample.y),axis=0)
    return new_sample
