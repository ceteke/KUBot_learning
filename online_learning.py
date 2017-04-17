import numpy as np
from data_handler import DataHandler
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from models import OnlineRegression
import math

class OnlineLearning():

    def __init__(self, data_set_size=-1):
        self.data_set_size = data_set_size
        self.dh = DataHandler()
        self.dh.collect_data(self.data_set_size)

    def scale(self, features):
        features = np.delete(features, 3)
        position = minmax_scale(features[0:3])
        position = np.multiply(position, 10.0)
        size = minmax_scale(features[3:6])
        histogram = minmax_scale(features[6:51])
        histogram = np.multiply(histogram, 0.1)
        new_feats = np.append(position, np.append(size, histogram))
        return new_feats

    def train(self):
        for a in self.dh.actions:
            a.split_train_test(0.2)
            for s in a.train_samples:
                if not np.array_equal(s.y, [-1.0]*52):
                    y_s = self.scale(s.y)
                else:
                    y_s = np.delete(s.y, 3)
                x_s = self.scale(s.X)
                o_min_distance = a.object_som.get_min_distance(x_s)
                if o_min_distance is None:
                    a.object_som.add_neuron(x_s)
                elif o_min_distance > 1:
                    new_cid = a.object_som.add_neuron(x_s)
                    a.obj_model_map[new_cid] = OnlineRegression()
                a.object_som.update(x_s)
                cluster_id = a.object_som.winner(x_s)
                a.obj_model_map[cluster_id].update(x_s, y_s)

                e_min_distance = a.effect_som.get_min_distance(y_s)
                if e_min_distance is None:
                    a.effect_som.add_neuron(y_s)
                elif e_min_distance > 1:
                    a.effect_som.add_neuron(y_s)
                a.effect_som.update(y_s)
            # a.gd.save(a.name)
            print len(a.object_som.weights)
            print len(a.effect_som.weights)
        for cid, model in a.obj_model_map.iteritems():
            plt.plot(model.Js, label=cid)
        plt.legend()
        plt.show()
        added = []
        for a in self.dh.actions:
            for s in a.test_samples:
                x_s = self.scale(s.X)
                y_s = self.scale(s.y)
                cluster_id = a.object_som.winner(x_s)
                model = a.obj_model_map[cluster_id]
                y_predicted = model.predict(x_s)
                b = np.subtract(y_s, y_predicted.flatten())
                J = np.matmul(b.T, b)
                print J / 2
                #print model.get_square_error(x_s, minmax_scale(s.y))
                if s.obj.id not in added:
                    print s.obj.id, a.effect_som.winner(y_predicted.flatten())
                    added.append(s.obj.id)

            for k, v in a.obj_model_map.iteritems():
                j_avgs = []
                t = 0.0
                for i in range(len(v.Js)):
                    t += v.Js[i]
                    j_avgs.append(t / float(i+1))
                #print min(j_avgs)
                plt.plot(j_avgs, label=k)
            plt.legend()
            plt.show()
