import pickle
from my_object import MyObject
import random
from sample import Sample
from models import SOM
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

class Action():

    def __init__(self, name):
        self.name = name
        self.objects = []
        self.samples = []

        self.effect_som = SOM(3, 0.01, 0.05, T1=100, T2=100)

        self.nn = Sequential()
        self.nn.add(Dense(128, input_dim=51, activation='relu'))
        self.nn.add(Dense(128, activation='relu'))
        self.nn.add(Dense(64, activation='relu'))
        self.nn.add(Dense(3, activation='relu'))
        self.nn.compile(loss='mean_absolute_error', optimizer='adagrad')

        self.object_scaler = MinMaxScaler()
        self.effect_scaler = MinMaxScaler()

        self.y_train_p = []
        self.y_test_p = []

    def save_models(self):
        self.nn.save('/Volumes/ROSDATA/models/nn.h5')
        pickle.dump(self.effect_som, open('/Volumes/ROSDATA/models/som.pkl', 'wb'))
        pickle.dump(self.object_scaler, open('/Volumes/ROSDATA/models/object_scaler.pkl', 'wb'))
        pickle.dump(self.effect_scaler, open('/Volumes/ROSDATA/models/effect_scaler.pkl', 'wb'))

    def load_models(self):
        self.nn = load_model('/Volumes/ROSDATA/models/push_nn_0.h5')
        self.effect_som = pickle.load(open('/Volumes/ROSDATA/models/push_effect_som_0.pkl', 'rb'))
        self.object_scaler = pickle.load(open('/Volumes/ROSDATA/models/object_scaler.pkl', 'rb'))
        self.effect_scaler = pickle.load(open('/Volumes/ROSDATA/models/effect_scaler.pkl', 'rb'))

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

        for s in self.train_samples:
            self.X_train.append(s.X)
            self.y_train.append(s.y)

        for s in self.test_samples:
            self.X_test.append(s.X)
            self.y_test.append(s.y)

    def scale_dataset(self):
        self.X_train = self.object_scaler.fit_transform(self.X_train)
        self.y_train = self.effect_scaler.fit_transform(self.y_train)
        self.X_test = self.object_scaler.transform(self.X_test)
        self.y_test = self.effect_scaler.transform(self.y_test)

        for y in self.y_train:
            self.y_train_p.append(y[0:3])

        for y in self.y_test:
            self.y_test_p.append(y[0:3])

    def __str__(self):
        return '%s: %s' % (self.name, [str(o) for o in self.objects])
