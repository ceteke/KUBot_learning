from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import pickle
from my_object import MyObject
import random
from sample import Sample

class Action():

    def __init__(self,name):
        self.name = name
        self.regressor = LinearRegression()
        self.effect_cluster = KMeans(n_clusters=2)
        self.online_effect_cluster = MiniBatchKMeans(n_clusters=2)
        self.gmm = GaussianMixture(n_components=2)
        self.objects = []
        self.clusters = {0:'Stay', 1:'Roll'}
        self.expected_effects = {'vcylinder1': 0,
                                'hcylinder1': 1,
                                'box1': 0,
                                'sphere1': 1,
                                'vcylinder0': 0,
                                'hcylinder0': 1,
                                'box0': 0,
                                'sphere0': 1}
        self.samples = []
        self.dimensions = 70 #69 + 1
        self.W = np.random.rand(self.dimensions,self.dimensions)
        self.Js = []

    def add_data(self, obj_name, obj_pose, X, y):
        obj_id = '%s%d' % (obj_name,obj_pose)
        obj = next((x for x in self.objects if x.id == obj_id), None)

        if obj is None:
            obj = MyObject(obj_name,obj_pose)
            self.objects.append(obj)

        sample = Sample(X, y, obj)
        sample.X /= np.max(np.abs(sample.X),axis=0)
        sample.y /= np.max(np.abs(sample.y),axis=0)
        self.samples.append(sample)

    def update_weights(self,x,y,alpha):
        J = self.get_square_error(x,y) / 2
        self.Js.append(J)
        dJdW = np.matmul(self.W,np.matmul(x,x.T)) - np.matmul(y,x.T)
        self.W -= alpha * dJdW

    def get_square_error(self, x, y):
        a = y - np.matmul(self.W, x)
        return np.matmul(a.T, a)[0][0]

    def get_gradient_descent_mse(self):
        total = 0.0
        c = 0.0
        for i in range(len(self.X_test)):
            x = self.X_test[i][np.newaxis].T
            y = self.y_test[i][np.newaxis].T
            x = np.vstack([x, [1.0]])
            y = np.vstack([y, [0.0]])
            err = self.get_square_error(x,y)
            total += err
            c += 1.0
        return total/c

    def preprocess(self, test_size):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        random.shuffle(self.samples)
        how_many = int(round(test_size*len(self.samples)))
        self.test_samples = self.samples[:how_many]
        self.train_samples = self.samples[how_many:]

        for s in self.train_samples:
            self.X_train.append(s.X)
            self.y_train.append(s.y)

        for s in self.test_samples:
            self.X_test.append(s.X)
            self.y_test.append(s.y)

    def offline_train(self):
        self.regressor.fit(self.X_train, self.y_train)
        self.gmm.fit(self.y_train)

    def get_cluster_accuracy(self):
        test_count = 0.0
        true_count = 0.0
        for s in self.test_samples:
            x = s.X.reshape(1,-1)
            y_predicted = self.regressor.predict(x)
            effect = self.gmm.predict(y_predicted)
            test_count += 1.0
            if self.expected_effects[s.obj.id] == effect[0]:
                true_count += 1.0
        return (true_count/test_count) * 100.0

    def get_regression_score(self):
        return self.regressor.score(self.X_test, self.y_test)

    def save(self,train_path):
        pickle.dump(self.regressor, open('%s%s_linear_regression'% (train_path, self.name), 'wb'))
        pickle.dump(self.gmm, open('%s%s_effect_cluster' % (train_path, self.name), 'wb'))

    def __str__(self):
        return '%s: %s' % (self.name, [str(o) for o in self.objects])
