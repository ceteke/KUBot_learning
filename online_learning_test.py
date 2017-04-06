import pickle
from sklearn.preprocessing import minmax_scale
from data_handler import DataHandler
import numpy as np
import math

dh = DataHandler(data_path='/media/cem/ROSDATA/ros_data/features/csv/')
dh.collect_data(-1)

for a in dh.actions:
    gd = pickle.load(open('/home/cem/learning/models/%s_gradient_descent' % (a.name),'rb'))
    a.split_train_test(1)
    print gd.get_rmse(a.X_test,a.y_test)
