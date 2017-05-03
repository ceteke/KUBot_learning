import numpy as np
from data_handler import DataHandler
import pyprind
import pprint

dh = DataHandler(data_path='/Volumes/ROSDATA/ros_data/features/test_data/')
dh.collect_data()

a = dh.actions[0]
a.load_models()

pp = pprint.PrettyPrinter(indent=2)
clusters = {}
errs = []

for s in a.samples:
    obj = s.obj
    x = a.object_scaler.transform(s.X.reshape(1, -1)).flatten()
    y = a.effect_scaler.transform(s.y.reshape(1, -1)).flatten()[0:3]

    y_predicted = a.nn.predict(x.reshape(1, -1)).flatten()
    err = np.linalg.norm(y_predicted-y)
    errs.append(err)

    obj_name = obj.name
    obj_pos = str(obj.pose)
    obj_id = obj.id

    if obj_id in dh.dropped:
        obj_pos += '*'

    if err > 0.1:
        obj_pos += 'H'

    cid = a.effect_som.winner(y_predicted)

    if cid in clusters:
        objs = clusters[cid]
        if obj_name in objs:
            objs[obj_name].append(obj_pos)
        else:
            objs[obj_name] = [obj_pos]
    else:
        clusters[cid] = {obj_name: [obj_pos]}

    print "Object:", obj.id
    print "Error:", err
    print "Cluster:", cid
    print "============================"

pp.pprint(clusters)
print "Mean errror:", np.mean(errs)
print "#neurons:", len(a.effect_som.weights)
print "Test size:", len(a.samples)