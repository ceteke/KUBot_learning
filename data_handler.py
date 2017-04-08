import os
import numpy as np
from action import Action

class DataHandler():

    def __init__(self, data_path='/Volumes/ROSDATA/ros_data/features/csv/'):
        self.data_path = data_path
        self.csv_folders = os.listdir(self.data_path)
        self.actions = []
        # self.dv = DataSetVisualization()

    def collect_data(self, set_size):
        print "Collecting data..."
        i = 0
        for csv_folder in self.csv_folders:
            if csv_folder[0] == '.':
                continue

            if i >= set_size and not set_size == -1:
                break

            object_info = csv_folder.split('_')
            object_name = object_info[2]
            object_pose = object_info[3]
            action_name = object_info[4]

            iteration_data_path = self.data_path + csv_folder
            before_csv = iteration_data_path + '/0_preproc.csv'
            after_csv = iteration_data_path + '/1_preproc.csv'

            before_features = np.genfromtxt(before_csv, delimiter=',')
            after_features = np.genfromtxt(after_csv, delimiter=',')
            effect_features = np.subtract(after_features, before_features)

            act = next((x for x in self.actions if x.name == action_name),
                       None)

            if act is None:
                act = Action(action_name)
                self.actions.append(act)


            act.add_data(object_name, int(object_pose), before_features,
                         effect_features)

            i += 1

        print "Collected actions: %s" % ([str(a) for a in self.actions])
