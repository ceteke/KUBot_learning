import os
import numpy as np
from action import Action

class DataHandler():

    def __init__(self, data_path='/Volumes/ROSDATA/ros_data/features/new2/'):
        self.data_path = data_path
        self.csv_folders = os.listdir(self.data_path)
        self.actions = []
        # self.dv = DataSetVisualization()

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
                        after_features = np.genfromtxt(after_csv, delimiter=',')
                        #print len(after_features)
                        if np.array_equal(after_features, np.array([-1.0]*51)):
                            continue
                        else:
                            effect_features = np.subtract(after_features, before_features)

                        act = next((x for x in self.actions if x.name == ad),
                                   None)

                        if act is None:
                            act = Action(ad)
                            self.actions.append(act)

                        act.add_data(od, int(id), before_features, effect_features)

                        # i += 1

        #print "Collected actions: %s" % ([str(a) for a in self.actions])
