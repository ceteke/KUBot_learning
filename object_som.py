from models import SOM, OnlineRegression
from data_handler import DataHandler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pprint

object_som = SOM(52, 0.2, 0.2, T1=1000, T2=1000)
dh = DataHandler()
dh.collect_data()
obj_model_map = {}
pp = pprint.PrettyPrinter(indent=2)


for a in dh.actions:
    a.split_train_test(0.1)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(a.X_train)
    y_train = y_scaler.fit_transform(a.y_train)
    X_test = x_scaler.transform(a.X_test)
    y_test = y_scaler.transform(a.y_test)

    for i in range(len(X_train)):
        x = X_train[i][3:51]
        y = y_train[i][0:3]
        o = a.train_samples[i].obj
        o_min_distance = object_som.get_min_distance(x)
        if o_min_distance > 0.4 or o_min_distance == -1:
            is_new = True
            new_cid = object_som.add_neuron(x)
            obj_model_map[new_cid] = OnlineRegression(alpha0=0.2)

        cluster_id = object_som.update(x)

        J = obj_model_map[cluster_id].update(x,y)

    clusters = {}
    for s in a.test_samples:
        obj_name = s.obj.name
        obj_pos = s.obj.pose
        x = s.X
        x_s = x_scaler.transform(x.reshape(1, -1)).flatten()[3:51]
        cid = object_som.winner(x_s)

        if cid in clusters:
            objs = clusters[cid]
            if obj_name in objs:
                objs[obj_name].append(obj_pos)
            else:
                objs[obj_name] = [obj_pos]
        else:
            clusters[cid] = {obj_name: [obj_pos]}

    pp.pprint(clusters)
    print len(object_som.weights)

    test_Js = {}
    all_Js = []
    for i in range(len(X_test)):
        x = X_test[i][3:51]
        y = y_test[i][0:3]
        predicted_cid = object_som.winner(x)
        regressor = obj_model_map[predicted_cid]
        err = regressor.get_error(x, y)
        #print err
        y_predicted = regressor.predict(x).flatten()
        #plt.plot(y, label='y')
        #plt.plot(y_predicted, label='y_p')
        #plt.legend()
        #plt.show()
        if predicted_cid in test_Js:
            test_Js[predicted_cid].append(err)
        else:
            test_Js[predicted_cid] = [err]
        all_Js.append(err)

    for k, v in test_Js.iteritems():
        print k, np.mean(v), len(v)
    print "Avg: ", np.mean(all_Js)
    print "#neurons: ", len(object_som.weights)
    #plt.legend()
    #plt.show()