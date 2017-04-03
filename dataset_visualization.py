import matplotlib.pyplot as plt
from offline_learning import OfflineLearning
from utils import scale_data_point
from sklearn.preprocessing import minmax_scale, scale, maxabs_scale


class DataSetVisualization():
    def predict_sample(self, action, sample):
        cid = action.gmm.predict(sample.y.reshape(1, -1))[0]
        print "%s: %d" % (sample.obj.id, cid)

    def plot_samples(self, samples):
        for l, s in samples.iteritems():
            plt.plot(s.y, label=l)
        plt.legend()
        plt.show()

    def plot_features(self, features):
        for l, f in features.iteritems():
            plt.plot(f, label=l)
        plt.legend()
        plt.show()

    def plot_cluster_samples(self, samples, model):
        for l, s in samples.iteritems():
            plt.plot(s.y, label=l)
        plt.plot(model.means_[0], label='STAY CLUSTER MEANS')
        plt.plot(model.means_[1], label='ROLL CLUSTER MEANS')
        plt.legend()
        plt.show()


ol = OfflineLearning()
best_trials = ol.get_best_fit(10, 0.2)
for best_trial in best_trials:
    print "Action: %s Regression Score: %f \
    Cluster Accuracy: %f" % (best_trial.action.name,
                             best_trial.regression_score,
                             best_trial.cluster_accuracy)
push = best_trials[0].action
box_pos1 = next((x for x in push.test_samples if x.obj.name == 'box'
                 and x.obj.pose == 1), None)

box_pos0 = next((x for x in push.test_samples if x.obj.name == 'box'
                 and x.obj.pose == 0), None)

sphere_pos0 = next((x for x in push.test_samples if x.obj.name == 'sphere'
                    and x.obj.pose == 0), None)

sphere_pos1 = next((x for x in push.test_samples if x.obj.name == 'sphere'
                    and x.obj.pose == 1), None)

vcylinder_pos0 = next((x for x in push.test_samples if x.obj.name == 'vcylinder'
                       and x.obj.pose == 0), None)

vcylinder_pos1 = next((x for x in push.test_samples if x.obj.name == 'vcylinder'
                       and x.obj.pose == 1), None)

hcylinder_pos0 = next((x for x in push.test_samples if x.obj.name == 'hcylinder'
                       and x.obj.pose == 0), None)

hcylinder_pos1 = next((x for x in push.test_samples if x.obj.name == 'hcylinder'
                       and x.obj.pose == 1), None)

dv = DataSetVisualization()
dv.plot_features({'box_pos1_my': minmax_scale((box_pos1.y)),
                  'box_pos1': box_pos1.y,
                  'box_pos1_scikit': push.effect_scaler.transform(box_pos1.y.reshape(1, -1)).reshape(-1, 1)}
                 )
