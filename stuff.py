import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale, scale
from online_learning import OnlineLearning
from copy import deepcopy


ol = OnlineLearning()

objs =  { 'hcylinder': [72, 765, 53, 47, 63, 46, 767, 769]}
x_scaler = pickle.load(open('/home/cem/learning/models/push_before_scaler.pkl', 'rb'))
y_scaler = pickle.load(open('/home/cem/learning/models/push_effect_scaler.pkl', 'rb'))
for o, rid in objs.iteritems():
    for r in rid:
        before_csv = '/media/cem/ROSDATA/ros_data/features/new6/652/push/%s/%d/0.csv' % (o, r)
        after_csv = '/media/cem/ROSDATA/ros_data/features/new6/652/push/%s/%d/1.csv' % (o, r)

        before_features = np.genfromtxt(before_csv, delimiter=',')
        after_features = np.genfromtxt(after_csv, delimiter=',')
        if np.array_equal(after_features, np.array([0.0] * 51)):
            print '%s_%d dropped' % (o, r)
            effect_features = deepcopy(before_features)
        else:
            effect_features = np.absolute(np.subtract(after_features, before_features))

        b = x_scaler.transform(before_features.reshape(1, -1)).flatten()[6:51]
        e = y_scaler.transform(effect_features.reshape(1, -1)).flatten()[0:3]

        plt.plot(b, label='%s_%d' % (o, r))


plt.legend()
plt.show()
