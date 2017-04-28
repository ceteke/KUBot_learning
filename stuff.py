import pickle
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import os

run_ids = [i for i in range(653, 660)]
objs = { 'box': [ '700*',
                 '196*',
                 '236*',
                 '374*',
                 '819*',
                 '351*',
                 '1007*',
                 '241*',
                 '401*',
                 '626',
                 '849*',
                 '1024',
                 '703*',
                 '567*',
                 '473*',
                 '845*'],
        'hcylinder': [ '23*',
                       '365*',
                       '196',
                       '423*',
                       '1374*',
                       '319*',
                       '1229*',
                       '159*',
                       '1390*',
                       '161*',
                       '299*',
                       '265',
                       '697',
                       '417*',
                       '541*'],
        'sphere': [ '90*',
                    '574*',
                    '1099*',
                    '1317*',
                    '496*',
                    '869*',
                    '1121*',
                    '269*',
                    '339*',
                    '153*',
                    '577',
                    '825*',
                    '889',
                    '794*',
                    '1021*',
                    '188*',
                    '840*',
                    '988*',
                    '275',
                    '283*',
                    '272*'],
        'vcylinder': [ '883*',
                       '992*',
                       '198',
                       '1005',
                       '268',
                       '525',
                       '558',
                       '296*',
                       '124*',
                       '350',
                       '515',
                       '491*',
                       '485*',
                       '970*',
                       '131*',
                       '441']}

x_scaler = pickle.load(open('/Users/Cem/learning/models/push_before_scaler.pkl', 'rb'))
y_scaler = pickle.load(open('/Users/Cem/learning/models/push_effect_scaler.pkl', 'rb'))
for o, rid in objs.iteritems():
    for r in rid:
        if '*' in r:
            r = r[:-1]
        for k in run_ids:
            before_csv = '/Volumes/ROSDATA/ros_data/features/new6/%d/push/%s/%s/0.csv' % (k, o, r)
            after_csv = '/Volumes/ROSDATA/ros_data/features/new6/%d/push/%s/%s/1.csv' % (k, o, r)
            if os.path.exists(before_csv):
                before_features = np.genfromtxt(before_csv, delimiter=',')
                after_features = np.genfromtxt(after_csv, delimiter=',')
                break
        if np.array_equal(after_features, np.array([0.0] * 51)):
            print '%s_%s dropped' % (o, r)
            effect_features = deepcopy(before_features)
        else:
            effect_features = np.absolute(np.subtract(after_features, before_features))

        b = x_scaler.transform(before_features.reshape(1, -1)).flatten()[6:51]
        e = y_scaler.transform(effect_features.reshape(1, -1)).flatten()[0:3]

        plt.plot([1,2,3], e, "o")
        plt.xlim([0.5,3.5])
        plt.xticks([0.5,1,2,3,3.5])


plt.legend()
plt.show()
