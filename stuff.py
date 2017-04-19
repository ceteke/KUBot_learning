import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale, scale


def myscale(features):
    if np.array_equal(features, np.array([-1.0] * 51)):
        return features
    position = features[0:3]
    others = features[3:51]
    position = np.multiply(position, 10.0)
    #others = np.multiply(others, 0.01)
    new_feats = np.append(position, minmax_scale(others))
    #new_feats = scale(new_feats)
    return new_feats

#obj_map = pickle.load(open('/home/cem/learning/models/push_map.pkl', 'rb'))

before_csv = '/Volumes/ROSDATA/ros_data/features/new2/626/push/hcylinder/26/0.csv'
after_csv = '/Volumes/ROSDATA/ros_data/features/new2/626/push/hcylinder/26/1.csv'
before_csv1 = '/Volumes/ROSDATA/ros_data/features/new2/626/push/vcylinder/0/0.csv'
after_csv1 = '/Volumes/ROSDATA/ros_data/features/new2/626/push/vcylinder/0/1.csv'

before_features = np.genfromtxt(before_csv, delimiter=',')
after_features = np.genfromtxt(after_csv, delimiter=',')
effect_features = np.absolute(np.subtract(after_features, before_features))

before_features1 = np.genfromtxt(before_csv1, delimiter=',')
after_features1 = np.genfromtxt(after_csv1, delimiter=',')
effect_features1 = np.absolute(np.subtract(after_features1, before_features1))
#print before_features[20]
#print before_features[19]
#print before_features[20] / before_features[19]
#plt.plot(before_features, label='before')
s_b = myscale(before_features1)
s_a = myscale(after_features1)
s_e = myscale(effect_features1)

b_b = myscale(before_features)
b_a = myscale(after_features)
b_e = myscale(effect_features)

#plt.plot(a_feats-b_feats, label='e1')
#plt.plot(s_b, label='sphere_b')
plt.plot(s_b, label='v_e')
#plt.plot(b_b, label='box_b')
plt.plot(b_b, label='h_e')
plt.legend()
plt.show()

#for k, v in obj_map.iteritems():
#    plt.plot(v.Js, label=k)
#plt.legend()
#plt.show()
