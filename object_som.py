from models import SOM, OnlineRegression
from data_handler import DataHandler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

object_som = SOM(51, 0.2, 0.2, T1=100, T2=100)
dh = DataHandler()
dh.collect_data()
obj_model_map = {}

for a in dh.actions:
    a.split_train_test(0.2)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(a.X_train)
    y_train = y_scaler.fit_transform(a.y_train)
    X_test = x_scaler.transform(a.X_test)
    y_test = y_scaler.transform(a.y_test)

    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]
        o_min_distance = object_som.get_min_distance(x)
        if o_min_distance > 2.4 or o_min_distance == -1:
            is_new = True
            new_cid = object_som.add_neuron(x)
            obj_model_map[new_cid] = OnlineRegression(alpha0=0.2, T=1000)

        cluster_id = object_som.update(x)
        obj_model_map[cluster_id].update(x, y)

    for cid, model in obj_model_map.iteritems():
        plt.plot(model.Js, label=cid)

    test_Js = {}
    all_Js = []
    for i in range(len(X_test)):
        x = X_test[i]
        y = y_test[i]
        predicted_cid = object_som.winner(x)
        regressor = obj_model_map[predicted_cid]
        err = regressor.get_error(x, y)
        print y, regressor.predict(x)
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