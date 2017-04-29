from models import SOM
from data_handler import DataHandler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.cluster import KMeans
import pprint
import pickle

pp = pprint.PrettyPrinter(indent=1)
effect_som = SOM(3, 0.01, 0.05, T1=100, T2=100)
#effect_aux = KMeans(n_clusters=10)
dh = DataHandler()
dh.collect_data()

for a in dh.actions:
    y_scaler = MinMaxScaler()
    a.split_train_test(0.1)
    y_train = y_scaler.fit_transform(a.y_train)

    for y in y_train:
        y = y[0:3]
        effect_min_distance = effect_som.get_min_distance(y)
        if effect_min_distance > 0.08 or effect_min_distance == -1:
            effect_som.add_neuron(y)
        else:
            effect_som.update(y)

    #effect_aux.fit(effect_som.weights)
    clusters = {}
    for s in a.test_samples:
        obj_name = s.obj.name
        obj_pos = str(s.obj.pose)
        obj_id = s.obj.id
        y = s.y
        if obj_id in dh.dropped:
            obj_pos += '*'
        y_s = y_scaler.transform(y.reshape(1, -1)).flatten()[0:3]
        cid = effect_som.winner(y_s.reshape(1,-1))

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

