import numpy as np
from data_handler import DataHandler
import matplotlib.pyplot as plt

class OnlineLearning():

    def __init__(self,data_set_size=-1):
        self.data_set_size = data_set_size
        self.dh = DataHandler()
        self.dh.collect_data(self.data_set_size)

    def train(self):
        for a in self.dh.actions:
            a.preprocess(0.2)
            for i in range(len(a.X_train)):
                x = a.X_train[i]
                y = a.y_train[i]
                x = x[np.newaxis].T
                y = y[np.newaxis].T
                x = np.vstack([x, [1.0]])
                y = np.vstack([y, [0.0]])
                is_interesting = a.update_weights(x,y,0.2)
                if not is_interesting:
                    break
            mse = a.get_gradient_descent_mse()
            print "MSE: %f" % (mse)
            plt.plot(a.Js)
            plt.ylabel('J')
            plt.show()
