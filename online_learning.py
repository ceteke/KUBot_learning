import numpy as np
from data_handler import DataHandler
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import math

class OnlineLearning():

    def __init__(self, data_set_size=-1):
        self.data_set_size = data_set_size
        self.dh = DataHandler()
        self.dh.collect_data(self.data_set_size)

    def train(self):
        for a in self.dh.actions:
            a.split_train_test(0.2)
            a.scale_dataset()
            for s in a.train_samples:
                y_s = minmax_scale(s.y)
                x_s = minmax_scale(s.X)
                a.gd.update(x_s, y_s)
                min_distance = a.som.get_min_distance(y_s)
                if min_distance > 1.25:
                    a.som.add_neuron(y_s)
                    # a.som.update(y_s)
                else:
                    a.som.update(y_s)
            # a.gd.save(a.name)
            print a.som.x
            rmse = a.gd.get_rmse(a.X_test, a.y_test)
            print "RMSE: %f" % (rmse)
        added = []
        for a in self.dh.actions:
            for s in a.test_samples:
                x_s = minmax_scale(s.X)
                y_predicted = a.gd.predict(x_s)
                if s.obj.id not in added:
                    print s.obj.id, a.som.winner(y_predicted.flatten())
                    added.append(s.obj.id)
