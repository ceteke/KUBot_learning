import numpy as np
from data_handler import DataHandler
import pyprind
import pprint

class OnlineLearning():

    def __init__(self, data_set_size=-1):
        self.data_set_size = data_set_size
        self.dh = DataHandler()
        self.dh.collect_data(self.data_set_size)

    def train(self):
        for a in self.dh.actions:
            a.split_train_test(0.01)
            a.scale_dataset()
            print a.name, len(a.X_train), len(a.X_test)
            #bar = pyprind.ProgBar(len(a.X_train), track_time=False, title='Training %s...' % (a.name))
            for i in range(len(a.X_train)):
                x = a.X_train[i]
                y = a.y_train_p[i]

                a.nn.fit(x.reshape(1, -1), y.reshape(1, -1), verbose=0, batch_size=1, epochs=10)
                e_min_distance = a.effect_som.get_min_distance(y)
                if e_min_distance == -1 or e_min_distance >= 0.08:
                    a.effect_som.add_neuron(y)

                a.effect_som.update(y)
                #bar.update()

            pp = pprint.PrettyPrinter(indent=2)
            clusters = {}
            for i in range(len(a.X_test)):
                x = a.X_test[i]
                y = a.y_test_p[i]
                obj = a.test_samples[i].obj
                y_predicted = a.nn.predict(x.reshape(1, -1)).flatten()
                err = np.linalg.norm(y-y_predicted)

                obj_name = obj.name
                obj_pos = str(obj.pose)
                obj_id = obj.id

                if obj_id in self.dh.dropped:
                    obj_pos += '*'

                if err > 0.05:
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

            print "#neurons:", len(a.effect_som.weights)



