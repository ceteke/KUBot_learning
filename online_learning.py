import numpy as np
from data_handler import DataHandler
import matplotlib.pyplot as plt
from models import OnlineRegression
import pprint
from sklearn.preprocessing import MinMaxScaler

class OnlineLearning():

    def __init__(self, data_set_size=-1):
        self.data_set_size = data_set_size
        self.dh = DataHandler()
        self.dh.collect_data(self.data_set_size)
        self.object_scaler = MinMaxScaler()
        self.effect_scaler = MinMaxScaler()
        for a in self.dh.actions:
            a.split_train_test(0.2)
            a.X_train = self.object_scaler.fit_transform(a.X_train)
            a.y_train = self.effect_scaler.fit_transform(a.y_train)
            a.X_test = self.object_scaler.transform(a.X_test)
            a.y_test = self.effect_scaler.transform(a.y_test)


    def train(self):
        pp = pprint.PrettyPrinter(indent=3)
        for a in self.dh.actions:
            for s in a.train_samples:
                y_s = s.y
                x_s = s.X
                o_min_distance = a.object_som.get_min_distance(x_s)
                if o_min_distance is None:
                    a.object_som.add_neuron(x_s)
                elif o_min_distance > 1000:
                    new_cid = a.object_som.add_neuron(x_s)
                    a.obj_model_map[new_cid] = OnlineRegression()
                else:
                    a.object_som.update(x_s)
                cluster_id = a.object_som.winner(x_s)
                a.obj_model_map[cluster_id].update(x_s, y_s)

                e_min_distance = a.effect_som.get_min_distance(y_s)
                if e_min_distance is None:
                    a.effect_som.add_neuron(y_s)
                elif e_min_distance > 1.2:
                    a.effect_som.add_neuron(y_s)
                else:
                    a.effect_som.update(y_s)
            # a.gd.save(a.name)

            a.effect_aux_som.fit(a.effect_som.weights)

            print "Object som #neurons:", len(a.object_som.weights)
            print "Effect som #neurons:", len(a.effect_som.weights)
            #print "Effect Aux. som #neurons:", len(a.effect_aux_som.weights)

        for cid, model in a.obj_model_map.iteritems():
            plt.plot(model.Js, label=cid)
        plt.legend()
        plt.show()
        added = []
        for a in self.dh.actions:
            clusters = {}
            for s in a.test_samples:
                x_s = s.X
                y_s = s.y
                cluster_id = a.object_som.winner(x_s)
                model = a.obj_model_map[cluster_id]
                y_predicted = model.predict(x_s)
                b = np.subtract(y_s, y_predicted.flatten())
                J = np.matmul(b.T, b)
                J = J / 2
                predicted_eid = a.effect_aux_som.predict(a.effect_som.get_winner_neuron(y_predicted.flatten()).reshape(1,-1))[0]
                if predicted_eid not in clusters:
                    clusters[predicted_eid] = [(s.obj.name, J, y_s, y_predicted.flatten(), cluster_id)]
                else:
                    clusters[predicted_eid].append((s.obj.name, J, y_s, y_predicted.flatten(), cluster_id))
            #print pp.pprint(clusters)
            correct_count = 0.0
            total = 0.0
            for k, v in clusters.iteritems():
                obj_count = {}
                y_s = []
                y_s_p = []
                J_total = 0.0
                J_count = 0.0
                for p_o in v:
                    y_s.append(p_o[2])
                    y_s_p.append(p_o[3])
                    J_total += p_o[1]
                    J_count += 1.0
                    if p_o[0] in obj_count:
                        obj_count[p_o[0]] += 1
                    else:
                        obj_count[p_o[0]] = 1
                print "Mean J", J_total/J_count
                avg_y = np.average(y_s, axis=0)
                avg_y_p = np.average(y_s_p, axis=0)
                print k, obj_count, avg_y, avg_y_p
                if 'sphere' in obj_count and 'box' in obj_count or 'box' in obj_count and 'hcylinder' in obj_count or 'vcylinder' in obj_count and 'sphere' in obj_count or 'vcylinder' in obj_count and 'hcylinder' in obj_count:
                    if not np.average(avg_y) < 0.09 or np.average(avg_y) < -0.5:
                        correct_count += 1.0
                if len(obj_count) == 1:
                    correct_count += 1.0
                total += 1.0
            print len(a.test_samples), len(a.train_samples)
            print (correct_count / total) * 100
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


