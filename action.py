from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
from my_object import MyObject
import random
from sample import Sample
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import MiniBatchKMeans
from models import OnlineRegression, SOM

class Action():

    def __init__(self, name):
        self.name = name
        self.regressor = LinearRegression()
        self.gmm = GaussianMixture(n_components=2)
        self.online_cluster = MiniBatchKMeans(n_clusters=2)
        self.objects = []
        self.clusters = {0: 'Stay', 1: 'Roll'}
        self.expected_effects = {'vcylinder1': 0,
                                 'hcylinder1': 1,
                                 'box1': 0,
                                 'sphere1': 1,
                                 'vcylinder0': 0,
                                 'hcylinder0': 1,
                                 'box0': 0,
                                 'sphere0': 1}
        self.samples = []
        self.gd = OnlineRegression()
        self.effect_som = SOM(51, 0.2, 0.0075)
        self.object_som = SOM(51, 0.2, 0.005)
        self.obj_model_map = {0: OnlineRegression()}

    def add_data(self, obj_name, obj_pose, X, y):
        obj_id = '%s%d' % (obj_name, obj_pose)
        obj = next((x for x in self.objects if x.id == obj_id), None)

        if obj is None:
            obj = MyObject(obj_name, obj_pose)
            self.objects.append(obj)

        sample = Sample(X, y, obj)
        self.samples.append(sample)

    def split_train_test(self, test_size):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        random.shuffle(self.samples)
        how_many = int(round(test_size*len(self.samples)))
        self.test_samples = self.samples[:how_many]
        self.train_samples = self.samples[how_many:]

    def scale_dataset(self):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        for s in self.train_samples:
            self.X_train.append(np.append(minmax_scale(s.X[0:7]), minmax_scale(s.X[7:52])))
            self.y_train.append(np.append(minmax_scale(s.y[0:7]), minmax_scale(s.y[7:52])))

        for s in self.test_samples:
            self.X_test.append(minmax_scale(s.X))
            self.y_test.append(minmax_scale(s.y))

    def offline_train(self):
        self.regressor.fit(self.X_train, self.y_train)
        self.gmm.fit(self.y_train)

    #DEPRECATED
    def get_cluster_accuracy(self):
        test_count = 0.0
        true_count = 0.0
        for i in range(len(self.test_samples)):
            s = self.test_samples[i]
            x = minmax_scale(s.X).reshape(1,-1)
            y_predicted = self.regressor.predict(x)
            effect = self.gmm.predict(y_predicted)[0]
            test_count += 1.0
            # print "%s: %d" % (s.obj.id, effect)
            if self.expected_effects[s.obj.id] == effect:
                true_count += 1.0
        return (true_count/test_count) * 100.0


    def get_regression_score(self):
        return self.regressor.score(self.X_test, self.y_test)

    def save(self, train_path):
        pickle.dump(self.regressor,
                    open('%s%s_linear_regression' % (train_path, self.name),
                         'wb'))
        pickle.dump(self.gmm, open('%s%s_effect_cluster' % (train_path,
                                                            self.name), 'wb'))

    def __str__(self):
        return '%s: %s' % (self.name, [str(o) for o in self.objects])
