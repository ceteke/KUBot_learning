from data_handler import DataHandler
from copy import deepcopy
from utils import scale_sample
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
import pickle
from offline_learning import OfflineLearning

ol = OfflineLearning()
best_trials = ol.get_best_fit(20,0.2)
for best_trial in best_trials:
    print "Action: %s Regression Score: %f Cluster Accuracy: %f" % (best_trial.action.name, best_trial.regression_score, best_trial.cluster_accuracy)
push = best_trials[0].action
box_pos1 = next((x for x in push.test_samples if x.obj.name == 'box' and x.obj.pose == 1), None)
box_pos1_scaled = scale_sample(box_pos1)
print "BOX 1: ", push.gmm.predict(box_pos1_scaled.y.reshape(1,-1))[0]
box_pos0 = next((x for x in push.test_samples if x.obj.name == 'box' and x.obj.pose == 0), None)
box_pos0_scaled = scale_sample(box_pos0)
#print "BOX 0: ", push.gmm.predict(box_pos0_scaled.y.reshape(1,-1))[0]
push.cluster_exclude_object(box_pos1.obj)
#plt.plot(box_pos1.y)
#plt.ylabel('Effect Value')
plt.plot(box_pos1_scaled.y, label='BOX POS1')
#plt.plot(box_pos0_scaled.y, label='BOX POS0')
stay_plt = plt.plot(push.gmm.means_[0], label='STAY CLUSTER MEANS')
roll_plt = plt.plot(push.gmm.means_[1], label='ROLL CLUSTER MEANS')
plt.legend()
plt.show()
