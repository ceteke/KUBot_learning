import numpy as np
import math
import pickle
from sklearn.preprocessing import minmax_scale

class SOM():

    def __init__(self, feature_size, alpha0, d0, T1=100, T2=1000, num_neurons=0):
        self.alpha0 = alpha0
        self.d0 = d0
        self.T1 = T1
        self.T2 = T2
        self.t = 0
        self.feature_size = feature_size
        self.weights = []
        self.num_neurons = num_neurons

    def init_weights(self):
        for _ in range(self.num_neurons):
            self.weights.append(np.random.rand(self.feature_size))

    def add_neuron(self, weight):
        self.weights.append(weight)
        return len(self.weights) - 1

    def decay_alpha(self):
        return self.alpha0 * np.exp(-1 * (self.t / self.T1))

    def decay_d(self):
        return self.d0 * np.exp(-1 * (self.t / self.T2))

    def get_bmu_index(self, x):
        diff = [np.linalg.norm(x - w) for w in self.weights]
        return np.argmin(diff)

    def get_min_distance(self, x):
        if len(self.weights) == 0:
            return -1
        diff = [np.linalg.norm(x - w) for w in self.weights]
        return np.min(diff)

    def winner(self, x):
        return self.get_bmu_index(x)

    def get_winner_neuron(self, x):
        return self.weights[self.winner(x)]

    def neighborhood(self, j, i):
        d = self.decay_d()
        weight_distance = np.linalg.norm(self.weights[i] - self.weights[j])
        if weight_distance > d:
            return 0.0
        return 1.0

    def update(self, x):
        i = self.get_bmu_index(x)
        for j in range(len(self.weights)):
            neighborhood = self.neighborhood(j, i)
            self.weights[j] = self.weights[j] + self.decay_alpha() * neighborhood * (x - self.weights[j])
        self.t += 1
        return i

class OnlineRegression():
    def __init__(self, dimensions = (4, 4), alpha0 = 0.00001, T=100):
        self.dimensions = dimensions
        self.alpha0 = alpha0
        self.W = np.random.rand(self.dimensions[0], self.dimensions[1])
        self.Js = []
        self.t = 0
        self.alpha0 = alpha0
        self.T = T

    def update(self, x, y):
        x_s = self.__preproc_x(x)
        y_s = y[np.newaxis].T
        y_s = np.vstack([y_s, [0.0]])
        J = self.get_square_error(x_s, y_s) / 2
        self.Js.append(J)
        dJdW = np.matmul(self.W, np.matmul(x_s, x_s.T)) - np.matmul(y_s, x_s.T)
        self.W -= self.decay_alpha() * dJdW
        self.t += 1

    def decay_alpha(self):
        return self.alpha0 * np.exp(-1 * (self.t / self.T))

    def get_square_error(self, x, y):
        a = y - np.matmul(self.W, x)
        return np.matmul(a.T, a)[0][0]

    def get_error(self, x, y):
        y_s = y[np.newaxis].T
        y_s = np.vstack([y_s, [0.0]])
        a = y_s - np.matmul(self.W, self.__preproc_x(x))
        return np.matmul(a.T, a)[0][0] / 2

    def predict(self, x):
        x_s = self.__preproc_x(x)
        return np.delete(np.matmul(self.W, x_s), self.dimensions[0]-1, 0).flatten()

    def __preproc_x(self, x):
        x_s = x[np.newaxis].T
        x_s = np.vstack([x_s, [1.0]])
        return x_s

    def get_rmse(self, X, y):
        total = 0.0
        c = 0.0
        for i in range(len(X)):
            x = X[i]
            y_s = y[i][np.newaxis].T
            y_predicted = self.predict(x)
            a = y_s - y_predicted
            err = np.matmul(a.T, a)[0][0]
            total += err
            c += 1.0
        return math.sqrt(total / c)

    def save(self, prefix):
        pickle.dump(self, open('/home/cem/learning/models/%s_gradient_descent' % (prefix), 'wb'))
