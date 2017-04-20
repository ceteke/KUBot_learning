import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale, scale
from online_learning import OnlineLearning


ol = OnlineLearning()

objs = {'hcylinder': [1180, 32], 'box': [10]}

for o, rid in objs.iteritems():
    for r in rid:
        before_csv = '/Volumes/ROSDATA/ros_data/features/new2/626/push/%s/%d/0.csv' % (o, r)
        after_csv = '/Volumes/ROSDATA/ros_data/features/new2/626/push/%s/%d/1.csv' % (o, r)

        before_features = np.genfromtxt(before_csv, delimiter=',')
        after_features = np.genfromtxt(after_csv, delimiter=',')
        effect_features = np.absolute(np.subtract(after_features, before_features))

        b = ol.object_scaler.transform(before_features.reshape(1, -1))
        e = ol.effect_scaler.transform(effect_features.reshape(1, -1))

        plt.plot(e.flatten(), label='%s_%d' % (o, r))


plt.legend()
plt.show()
