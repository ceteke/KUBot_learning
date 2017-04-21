from models import SOM, OnlineRegression
from data_handler import DataHandler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

object_som = SOM(48, 0.2, 0.2, T1=100, T2=100)
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
        x_h = X_train[i][3:51]
        x_p = X_train[i][0:3]
        y = y_train[i][0:3]
        o_min_distance = object_som.get_min_distance(x_h)
        if o_min_distance > 5 or o_min_distance == -1:
            is_new = True
            new_cid = object_som.add_neuron(x_h)
            obj_model_map[new_cid] = OnlineRegression(alpha0=0.2, T=1000)

        cluster_id = object_som.update(x_h)
        obj_model_map[cluster_id].update(x_p, y)

    test_Js = {}
    all_Js = []
    for i in range(len(X_test)):
        x_h = X_test[i][3:51]
        x_p = X_test[i][0:3]
        y = y_test[i][0:3]
        predicted_cid = object_som.winner(x_h)
        regressor = obj_model_map[predicted_cid]
        err = regressor.get_error(x_p, y)
        #print err
        y_predicted = regressor.predict(x_p)
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