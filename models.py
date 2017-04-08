import numpy as np
import math
import pickle
from sklearn.preprocessing import minmax_scale

class SOM():

    def __init__(self, x, y, feature_size, alpha0, sigma0, T1=5, T2=5):
        self.x = x # Num of columns
        self.y = y # Num of rows
        self.alpha0 = alpha0
        self.sigma0 = sigma0
        self.T1 = T1
        self.T2 = T2
        self.t = 0
        self.feature_size = feature_size
        self.init_weights()

    def init_weights(self):
        self.weights = []
        for i in range(self.x * self.y):
            self.weights.append(np.random.rand(self.feature_size))

    def add_neuron(self, weight):
        # Add only to x axis
        self.weights.append(weight)
        self.x += 1

    def decay_alpha(self):
        return self.alpha0 * np.exp(-1 * (self.t / self.T1))

    def decay_sigma(self):
        return self.sigma0 * np.exp(-1 * (self.t / self.T2))

    def get_bmu_index(self, x):
        diff = [np.linalg.norm(x - w) for w in self.weights]
        return np.argmin(diff)

    def get_min_distance(self, x):
        diff = [np.linalg.norm(x - w) for w in self.weights]
        return np.min(diff)

    def winner(self, x):
        return np.unravel_index(self.get_bmu_index(x), (self.y, self.x))

    def distance(self, j, i):
        coordinates = np.unravel_index([j, i], (self.y, self.x))
        return np.linalg.norm(coordinates[0] - coordinates[1])

    def neighborhood(self, j, i):
        d = self.distance(j, i)
        return np.exp(-1 * (d**2/(2*(self.decay_sigma()**2))))

    def update(self, x):
        i = self.get_bmu_index(x)
        for j in range(len(self.weights)):
            neighborhood = self.neighborhood(j, i)
            self.weights[j] = self.weights[j] + self.decay_alpha() * neighborhood * (x - self.weights[j])
        self.t += 1

    def quantization(self, data):
        q = zeros(data.shape)
        for i, x in enumarate(data):
            q[i] = self.weights[self.get_bmu_index(x)]
        return q

class GradientDescent():

    def __init__(self, scaler, dimensions = 70, alpha0 = 0.2):
        self.dimensions = dimensions
        self.alpha0 = alpha0
        self.W = np.random.rand(self.dimensions, self.dimensions)
        self.Js = []
        self.t = 0
        self.scaler = scaler
        self.alpha_t = self.alpha0

    def update(self, x, y):
        x_s = self.__preproc_x(x)
        y_s = self.scaler(y)
        y_s = y_s[np.newaxis].T
        y_s = np.vstack([y_s, [0.0]])
        J = self.get_square_error(x_s, y_s) / 2
        self.Js.append(J)
        dJdW = np.matmul(self.W, np.matmul(x_s, x_s.T)) - np.matmul(y_s, x_s.T)
        self.W -= self.alpha_t * dJdW
        alpha_t = self.alpha0*500/(self.t+500)
        self.t += 1

    def get_square_error(self, x, y):
        a = y - np.matmul(self.W, x)
        return np.matmul(a.T, a)[0][0]

    def predict(self, x):
        x_s = self.__preproc_x(x)
        return np.delete(np.matmul(self.W, x_s), self.dimensions-1, 0)

    def __preproc_x(self, x):
        x_s = self.scaler(x)
        x_s = x_s[np.newaxis].T
        x_s = np.vstack([x_s, [1.0]])
        return x_s

    def get_rmse(self, X, y):
        total = 0.0
        c = 0.0
        for i in range(len(X)):
            x = X[i]
            y_s = self.scaler(y[i])
            y_s = y_s[np.newaxis].T
            y_predicted = self.predict(x)
            a = y_s - y_predicted
            err = np.matmul(a.T, a)[0][0]
            total += err
            c += 1.0
        return math.sqrt(total / c)

    def save(self, prefix):
        pickle.dump(self, open('/home/cem/learning/models/%s_gradient_descent' % (prefix), 'wb'))