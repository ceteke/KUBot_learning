from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pickle
from my_object import MyObject

class Action():

    def __init__(self,name):
        self.name = name
        self.regressor = LinearRegression()
        self.before_scaler = MinMaxScaler()
        self.effect_scaler = MinMaxScaler()
        self.effect_cluster = KMeans(n_clusters=2)
        self.objects = []

    def add_data_to_object(self, obj_name, obj_pose, X, y):
        obj_id = '%s%d' % (obj_name,obj_pose)
        obj = next((x for x in self.objects if x.id == obj_id), None)

        if obj is None:
            obj = MyObject(obj_name,obj_pose)
            self.objects.append(obj)

        obj.X.append(X)
        obj.y.append(y)

    def train(self):
        X_all = []
        y_all = []

        for o in self.objects:
            X_all += o.X
            y_all += o.y

        print "Training action: %s\nSet size: %d" % (self.name, len(X_all))
        print "Objects: %s" % ([str(o) for o in self.objects])

        X_all = self.before_scaler.fit_transform(X_all)
        y_all = self.effect_scaler.fit_transform(y_all)

        self.regressor.fit(X_all, y_all)
        self.effect_cluster.fit(y_all)

    def save(self,train_path):
        pickle.dump(self.regressor, open('%s%s_linear_regression'% (train_path, self.name), 'wb'))
        pickle.dump(self.before_scaler, open('%s%s_before_scaler' % (train_path, self.name), 'wb'))
        pickle.dump(self.effect_scaler, open('%s%s_effect_scaler' % (train_path, self.name), 'wb'))
        pickle.dump(self.effect_cluster, open('%s%s_effect_cluster' % (train_path, self.name), 'wb'))

    def __str__(self):
        return self.name
