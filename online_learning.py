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
            for s in a.train_samples:
                a.gd.update(s.X, s.y)
                min_distance = a.som.get_min_distance(s.y)
                if min_distance > 20:
                    a.som.add_neuron(s.y)
                    # a.som.update(s.y)
                else:
                    a.som.update(s.y)
            # a.gd.save(a.name)
            print a.som.x
            rmse = a.gd.get_rmse(a.X_test, a.y_test)
            print "RMSE: %f" % (rmse)
        for a in self.dh.actions:
            for s in a.test_samples:
                y_predicted = a.gd.predict(s.X)
                print s.obj.id, a.som.winner(y_predicted.flatten())
