import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale, scale


def scale(features):
    features = np.delete(features, 3)
    position = minmax_scale(features[0:3])
    position = np.multiply(position, 10.0)
    size = minmax_scale(features[3:6])
    histogram = minmax_scale(features[6:51])
    new_feats = np.append(position, np.append(size, histogram))
    return new_feats

#obj_map = pickle.load(open('/home/cem/learning/models/push_map.pkl', 'rb'))

before_csv = '/media/cem/ROSDATA/ros_data/features/new1/620/push/hcylinder/4/0.csv'
after_csv = '/media/cem/ROSDATA/ros_data/features/new1/620/push/hcylinder/4/1.csv'
before_csv1 = '/media/cem/ROSDATA/ros_data/features/new1/620/push/vcylinder/47/0.csv'
after_csv1 = '/media/cem/ROSDATA/ros_data/features/new1/620/push/vcylinder/47/1.csv'

before_features = np.genfromtxt(before_csv, delimiter=',')
after_features = np.genfromtxt(after_csv, delimiter=',')
effect_features = np.absolute(np.subtract(after_features, before_features))

before_features1 = np.genfromtxt(before_csv1, delimiter=',')
after_features1 = np.genfromtxt(after_csv1, delimiter=',')
effect_features1 = np.subtract(after_features1, before_features1)
#print before_features[20]
#print before_features[19]
#print before_features[20] / before_features[19]
#plt.plot(before_features, label='before')
s_b = scale(before_features1)
s_a = scale(after_features1)
s_e = scale(effect_features1)

b_b = scale(before_features)
b_a = scale(after_features)
b_e = scale(effect_features)

#plt.plot(a_feats-b_feats, label='e1')
#plt.plot(s_b, label='sphere_b')
plt.plot(s_e, label='v_e')
#plt.plot(b_b, label='box_b')
plt.plot(b_e, label='h_e')

plt.legend()
plt.show()

#for k, v in obj_map.iteritems():
#    plt.plot(v.Js, label=k)
#plt.legend()
#plt.show()
