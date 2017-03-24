from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
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
        self.before_scaler = MinMaxScaler()
        self.effect_scaler = MinMaxScaler()
        self.effect_cluster = KMeans(n_clusters=2)
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

    def add_data(self, obj_name, obj_pose, X, y):
        obj_id = '%s%d' % (obj_name,obj_pose)
        obj = next((x for x in self.objects if x.id == obj_id), None)

        if obj is None:
            obj = MyObject(obj_name,obj_pose)
            self.objects.append(obj)

        sample = Sample(X, y, obj)
        self.samples.append(sample)

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

        #print "Preprocessing action: %s\nSet size: %d\nTrain set size: %d\nTest set size: %d" % (self.name, len(self.samples), len(self.X_train),len(self.X_test))
        #print "Objects: %s" % ([str(o) for o in self.objects])

        self.X_train = self.before_scaler.fit_transform(self.X_train)
        self.X_test = self.before_scaler.transform(self.X_test)
        self.y_train = self.effect_scaler.fit_transform(self.y_train)
        self.y_test = self.effect_scaler.transform(self.y_test)

    def train(self):
        self.regressor.fit(self.X_train, self.y_train)
        self.gmm.fit(self.y_train)

    def get_cluster_accuracy(self):
        test_count = 0.0
        true_count = 0.0
        for s in self.test_samples:
            x = s.X.reshape(1,-1)
            x = self.before_scaler.transform(x)
            y_predicted = self.regressor.predict(x)
            effect = self.gmm.predict(y_predicted)[0]
            test_count += 1.0
            if self.expected_effects[s.obj.id] == effect:
                true_count += 1.0
        return (true_count/test_count) * 100.0

    def get_regression_score(self):
        return self.regressor.score(self.X_test, self.y_test)

    def save(self,train_path):
        pickle.dump(self.regressor, open('%s%s_linear_regression'% (train_path, self.name), 'wb'))
        pickle.dump(self.before_scaler, open('%s%s_before_scaler' % (train_path, self.name), 'wb'))
        pickle.dump(self.effect_scaler, open('%s%s_effect_scaler' % (train_path, self.name), 'wb'))
        pickle.dump(self.effect_cluster, open('%s%s_effect_cluster' % (train_path, self.name), 'wb'))

    def __str__(self):
        return '%s: %s' % (self.name, [str(o) for o in self.objects])
