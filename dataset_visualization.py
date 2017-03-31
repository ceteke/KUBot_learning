from data_handler import DataHandler
from copy import deepcopy
from utils import scale_sample, z_score_sample_scale, zero_one_scaler
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
import pickle
from offline_learning import OfflineLearning


class DataSetVisualization():
    def predict_sample(self,action,sample):
        cid = action.gmm.predict(sample.y.reshape(1,-1))[0]
        print "%s: %d" % (sample.obj.id, cid)

    def plot_samples(self,samples):
        for l,s in samples.iteritems():
            plt.plot(s.y, label=l)
        plt.legend()
        plt.show()

    def plot_clusters_samples(self,samples,model):
        for s in samples:
            plt.plot(s.y, label=s.obj.id)
        plt.plot(model.means_[0], label='STAY CLUSTER MEANS')
        plt.plot(model.means_[1], label='ROLL CLUSTER MEANS')
        plt.legend()
        plt.show()

ol = OfflineLearning()
best_trials = ol.get_best_fit(10,0.2)
for best_trial in best_trials:
    print "Action: %s Regression Score: %f Cluster Accuracy: %f" % (best_trial.action.name, best_trial.regression_score, best_trial.cluster_accuracy)
push = best_trials[0].action
box_pos1 = next((x for x in push.test_samples if x.obj.name == 'box' and x.obj.pose == 1), None)
box_pos1_scaled = scale_sample(box_pos1)
box_pos1_scaled2 = zero_one_scaler(box_pos1)

box_pos0 = next((x for x in push.test_samples if x.obj.name == 'box' and x.obj.pose == 0), None)
box_pos0_scaled = scale_sample(box_pos0)

sphere_pos0 = next((x for x in push.test_samples if x.obj.name == 'sphere' and x.obj.pose == 0), None)
sphere_pos0_scaled = scale_sample(sphere_pos0)

sphere_pos1 = next((x for x in push.test_samples if x.obj.name == 'sphere' and x.obj.pose == 1), None)
sphere_pos1_scaled = scale_sample(sphere_pos1)

dv = DataSetVisualization()
dv.predict_sample(push,box_pos0_scaled)
dv.predict_sample(push,box_pos1_scaled)
dv.predict_sample(push,sphere_pos1_scaled)
dv.predict_sample(push,sphere_pos0_scaled)
dv.plot_samples({'eski': box_pos1_scaled, 'yeni': box_pos1_scaled2})
