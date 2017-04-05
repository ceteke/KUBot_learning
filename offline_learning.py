from data_handler import DataHandler
from trial import Trial


class OfflineLearning():

    def __init__(self, data_set_size=-1):
        self.data_set_size = data_set_size
        self.dh = DataHandler(data_path='/media/cem/ROSDATA/ros_data/features/csv/')
        self.dh.collect_data(self.data_set_size)

    def get_best_fit(self, n_try, test_size):
        best_trials = []
        for a in self.dh.actions:
            trials = []
            for i in range(n_try):
                a.split_train_test(test_size)
                a.scale_dataset()
                a.offline_train()
                trials.append(Trial(a, a.get_regression_score()))
            # print trials
            trials.sort(key=lambda t: (t.regression_score), reverse=True)
            #trials.sort(key=lambda t: t.cluster_accuracy, reverse=True)
            best_trials.append(trials[0])
        return best_trials
