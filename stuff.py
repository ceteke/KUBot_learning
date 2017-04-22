import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale, scale
from online_learning import OnlineLearning
from copy import deepcopy


ol = OnlineLearning()

objs = { 'box': [ 935,
                 885,
                 929,
                 1437,
                 820,
                 1511,
                 512,
                 1275,
                 1698,
                 1472,
                 1041,
                 858,
                 1114,
                 1125,
                 1775,
                 873]}
for o, rid in objs.iteritems():
    for r in rid:
        before_csv = '/Volumes/ROSDATA/ros_data/features/new2/626/push/%s/%d/0.csv' % (o, r)
        after_csv = '/Volumes/ROSDATA/ros_data/features/new2/626/push/%s/%d/1.csv' % (o, r)

        before_features = np.genfromtxt(before_csv, delimiter=',')
        after_features = np.genfromtxt(after_csv, delimiter=',')
        if np.array_equal(after_features, np.array([-1.0] * 51)):
            print '%s_%d dropped' % (o, r)
            effect_features = deepcopy(before_features)
        else:
            effect_features = np.absolute(np.subtract(after_features, before_features))

        b = ol.object_scaler.transform(before_features.reshape(1, -1)).flatten()[3:51]
        e = ol.effect_scaler.transform(effect_features.reshape(1, -1)).flatten()[0:3]

        plt.plot(b, label='%s_%d' % (o, r))


plt.legend()
plt.show()
