import numpy as np
from data_handler import DataHandler
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import math

class OnlineLearning():

    def __init__(self, data_set_size=-1):
        self.data_set_size = data_set_size
        self.dh = DataHandler(data_path='/media/cem/ROSDATA/ros_data/features/csv/')
        self.dh.collect_data(self.data_set_size)

    def train(self):
        for a in self.dh.actions:
            a.split_train_test(0.2)
            for i in range(len(a.X_train)):
                a.gd.update(a.X_train[i], a.y_train[i])
            rmse = a.gd.get_rmse(a.X_test, a.y_test)
            print "RMSE: %f" % (rmse)
            plt.plot(a.gd.Js)
            plt.ylabel('J')
            plt.show()
