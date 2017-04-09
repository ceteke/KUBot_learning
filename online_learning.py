import numpy as np
from data_handler import DataHandler
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from models import GradientDescent
import math

class OnlineLearning():

    def __init__(self, data_set_size=-1):
        self.data_set_size = data_set_size
        self.dh = DataHandler()
        self.dh.collect_data(self.data_set_size)

    def train(self):
        for a in self.dh.actions:
            a.split_train_test(0.2)
            a.scale_dataset()
            for s in a.train_samples:
                y_s = minmax_scale(s.y)
                x_s = minmax_scale(s.X)

                o_min_distance = a.object_som.get_min_distance(x_s)
                if o_min_distance is None:
                    a.object_som.add_neuron(x_s)
                elif o_min_distance > 0.8:
                    new_cid = a.object_som.add_neuron(x_s)
                    a.obj_model_map[new_cid] = GradientDescent()
                cluster_id = a.object_som.winner(x_s)[1]
                a.obj_model_map[cluster_id].update(x_s, y_s)

                e_min_distance = a.effect_som.get_min_distance(y_s)
                if e_min_distance is None:
                    a.effect_som.add_neuron(y_s)
                elif e_min_distance > 1.25:
                    a.effect_som.add_neuron(y_s)
                a.effect_som.update(y_s)
            # a.gd.save(a.name)
            print a.object_som.x
            print a.effect_som.x
        for cid, model in a.obj_model_map.iteritems():
            plt.plot(model.Js, label=cid)
        plt.legend()
        plt.show()
        added = []
        for a in self.dh.actions:
            for s in a.test_samples:
                x_s = minmax_scale(s.X)
                cluster_id = a.object_som.winner(x_s)[1]
                model = a.obj_model_map[cluster_id]
                y_predicted = model.predict(x_s)
                #print model.get_square_error(x_s, minmax_scale(s.y))
                if s.obj.id not in added:
                    print s.obj.id, a.effect_som.winner(y_predicted.flatten())[1]
                    added.append(s.obj.id)
