from data_handler import DataHandler
import copy
from trial import Trial

class OfflineLearning():

    def __init__(self,data_set_size=-1):
        self.data_set_size = data_set_size
        self.dh = DataHandler()
        self.dh.collect_data(self.data_set_size)

    def get_best_fit(self, n_try, test_size):
        best_trials = []
        for a in self.dh.actions:
            trials = []
            for i in range(n_try):
                a_clone = copy.deepcopy(a)
                a.preprocess(test_size)
                a.train()
                trials.append(Trial(a,a.get_regression_score(), a.get_cluster_accuracy()))
            trials.sort(key=lambda t: t.regression_score, reverse=True)
            trials.sort(key=lambda t: t.cluster_accuracy, reverse=True)
            best_trials.append(trials[0])
        return best_trials
