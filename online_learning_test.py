import pickle
from sklearn.preprocessing import minmax_scale
from data_handler import DataHandler
import numpy as np
import math

dh = DataHandler(data_path='/media/cem/ROSDATA/ros_data/features/csv/')
dh.collect_data(-1)

for a in dh.actions:
    W = pickle.load(open('/home/cem/learning/models/%s_weights' % (a.name),'rb'))
    c = 0.0
    total = 0.0
    for s in a.samples:
        X = minmax_scale(s.X)
        y = minmax_scale(s.y)
        X = X[np.newaxis].T
        y = y[np.newaxis].T
        X = np.vstack([X, [1.0]])
        a = y - np.delete(np.matmul(W, X), 69, 0)
        err = np.matmul(a.T, a)[0][0]
        total += err
        c += 1.0
    print math.sqrt(total / c), 'RMSE'
