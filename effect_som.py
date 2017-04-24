from models import SOM
from data_handler import DataHandler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import KMeans
import pprint
import pickle

pp = pprint.PrettyPrinter(indent=2)
effect_som = SOM(3, 0.1, 0.03, T1=100, T2=100)
aux_effect_cluster = KMeans(n_clusters=10)
dh = DataHandler()
dh.collect_data()

for a in dh.actions:
    y_scaler = pickle.load(open('/Users/Cem/learning/models/push_effect_scaler.pkl', 'rb'))
    a.split_train_test(0.2)
    y_train = y_scaler.fit_transform(a.y_train)

    for y in y_train:
        y = y[0:3]
        effect_min_distance = effect_som.get_min_distance(y)
        if effect_min_distance > 0.08 or effect_min_distance == -1:
            effect_som.add_neuron(y)

        effect_som.update(y)

    clusters = {}
    for s in a.test_samples:
        obj_name = s.obj.name
        obj_pos = s.obj.pose
        y = s.y
        y_s = y_scaler.transform(y.reshape(1, -1)).flatten()[0:3]
        cid = effect_som.winner(y_s)

        if cid in clusters:
            objs = clusters[cid]
            if obj_name in objs:
                objs[obj_name].append(obj_pos)
            else:
                objs[obj_name] = [obj_pos]
        else:
            clusters[cid] = {obj_name: [obj_pos]}

    pp.pprint(clusters)
    print len(effect_som.weights)

