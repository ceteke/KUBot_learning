import numpy as np
from data_handler import DataHandler
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from models import OnlineRegression
import math
import pprint

class OnlineLearning():

    def __init__(self, data_set_size=-1):
        self.data_set_size = data_set_size
        self.dh = DataHandler()
        self.dh.collect_data(self.data_set_size)

    def scale(self, features):
        if np.array_equal(features, np.array([-1.0]*52)):
            return np.array([-1.0]*51)
        features = np.delete(features, 3)
        f_s = minmax_scale(features)
        position = f_s[0:3]
        others = f_s[3:51]
        others = np.multiply(others, 0.01)
        new_feats = np.append(position, others)
        return new_feats

    def train(self):
        pp = pprint.PrettyPrinter(indent=3)
        for a in self.dh.actions:
            a.split_train_test(0.1)
            for s in a.train_samples:
                y_s = self.scale(s.y)
                x_s = self.scale(s.X)
                o_min_distance = a.object_som.get_min_distance(x_s)
                if o_min_distance is None:
                    a.object_som.add_neuron(x_s)
                elif o_min_distance > 0.01:
                    new_cid = a.object_som.add_neuron(x_s)
                    a.obj_model_map[new_cid] = OnlineRegression()
                else:
                    a.object_som.update(x_s)
                cluster_id = a.object_som.winner(x_s)
                a.obj_model_map[cluster_id].update(x_s, y_s)

                e_min_distance = a.effect_som.get_min_distance(y_s)
                if e_min_distance is None:
                    a.effect_som.add_neuron(y_s)
                elif e_min_distance > 0.03:
                    a.effect_som.add_neuron(y_s)
                else:
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
            clusters = {}
            for s in a.test_samples:
                x_s = self.scale(s.X)
                y_s = self.scale(s.y)
                cluster_id = a.object_som.winner(x_s)
                model = a.obj_model_map[cluster_id]
                y_predicted = model.predict(x_s)
                b = np.subtract(y_s, y_predicted.flatten())
                J = np.matmul(b.T, b)
                J = J / 2
                predicted_eid = a.effect_som.winner(y_predicted.flatten())
                if predicted_eid not in clusters:
                    clusters[predicted_eid] = [(s.obj.name, J)]
                else:
                    clusters[predicted_eid].append((s.obj.name, J))
            print pp.pprint(clusters)

            for k, v in clusters.iteritems():
                obj_count = {}
                for p_o in v:
                    if p_o[0] in obj_count:
                        obj_count[p_o[0]] += 1
                    else:
                        obj_count[p_o[0]] = 1
                print k, obj_count

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
