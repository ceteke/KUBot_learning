import os
import numpy as np
from action import Action
from copy import deepcopy

class DataHandler():

    def __init__(self, data_path='/Volumes/ROSDATA/ros_data/features/new4/'):
        self.data_path = data_path
        self.csv_folders = os.listdir(self.data_path)
        self.actions = []
        # self.dv = DataSetVisualization()

    def reduce_features(self, features):
        preproc_features = features[0:3]
        histogram = features[6:51]

        hist_ranges = {15: 5}
        hist_order = [15, 15, 15]
        crr_index = 0

        for ho in hist_order:
            step = hist_ranges[ho]
            preproc = [np.sum(histogram[crr_index + i * step:crr_index + step * (i + 1)]) for i in range(ho / step)]
            preproc_features = np.append(preproc_features, preproc)
            crr_index += ho

        return np.array(preproc_features)

    def collect_data(self, set_size = -1):
        print "Collecting data..."
        # i = 0
        for csv_folder in self.csv_folders:
            if csv_folder[0] == '.':
                continue

            # if i >= set_size and not set_size == -1:
            #    break

            run_directory = os.path.join(self.data_path, csv_folder)
            action_directorties = os.listdir(run_directory)

            for ad in action_directorties:
                if ad[0] == '.':
                    continue

                action_directory = os.path.join(run_directory, ad)

                object_directories = os.listdir(action_directory)

                for od in object_directories:
                    if od[0] == '.':
                        continue

                    object_directory = os.path.join(action_directory, od)

                    iteration_directories = os.listdir(object_directory)

                    for id in iteration_directories:
                        if id[0] == '.':
                            continue

                        iteration_directory = os.path.join(object_directory, id)

                        before_csv = iteration_directory + '/0.csv'
                        after_csv = iteration_directory + '/1.csv'

                        before_features = np.genfromtxt(before_csv, delimiter=',')
                        #before_features = self.reduce_features(before_features)
                        after_features = np.genfromtxt(after_csv, delimiter=',')
                        #after_features = self.reduce_features(after_features)
                        #print len(after_features)
                        if np.array_equal(after_features, np.array([-1.0]*51)):
                            effect_features = np.absolute(deepcopy(before_features))
                        else:
                            effect_features = np.absolute(np.subtract(after_features, before_features))

                        act = next((x for x in self.actions if x.name == ad),
                                   None)

                        if act is None:
                            act = Action(ad)
                            self.actions.append(act)

                        act.add_data(od, int(id), before_features, effect_features)

                        # i += 1

        #print "Collected actions: %s" % ([str(a) for a in self.actions])
