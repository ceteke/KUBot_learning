import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale, scale
from online_learning import OnlineLearning
from copy import deepcopy


ol = OnlineLearning()

objs =  { 'box': [ 688,
                1483,
                565,
                1503,
                393,
                1517,
                1585,
                1125,
                664,
                391,
                25,
                1244,
                1270,
                977,
                326,
                573,
                1477,
                1437,
                1582,
                719,
                523,
                816,
                1237,
                302,
                609,
                1162,
                23,
                1068,
                1466,
                1179,
                802,
                242,
                777,
                850,
                857,
                815,
                46,
                501,
                1018,
                432,
                843,
                1511,
                1092,
                291,
                1090,
                1638,
                1636,
                124,
                1462,
                372,
                139,
                724,
                512,
                1013],
       'hcylinder': [988, 353, 1456, 272, 355, 202],
       'sphere': [172],
       'vcylinder': [ 1240,
                      546,
                      1391,
                      538,
                      741,
                      536,
                      602,
                      694,
                      1390,
                      1384,
                      86,
                      397,
                      1044,
                      1676,
                      1591,
                      72,
                      552,
                      209,
                      483,
                      88,
                      106,
                      716,
                      231,
                      748,
                      1507,
                      99,
                      903,
                      307,
                      742,
                      316,
                      599,
                      569,
                      1238,
                      1422,
                      1367,
                      1356,
                      1271,
                      1423,
                      415,
                      684,
                      1761,
                      205,
                      87,
                      317,
                      156,
                      1701,
                      1343,
                      98,
                      1471,
                      567,
                      606,
                      682,
                      886,
                      1599,
                      503,
                      1081,
                      313,
                      406,
                      1198,
                      101,
                      1245,
                      155,
                      1578,
                      421,
                      1241,
                      1075,
                      974,
                      1382]}


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

        b = ol.object_scaler.transform(before_features.reshape(1, -1))
        e = ol.effect_scaler.transform(effect_features.reshape(1, -1)).flatten()[0:3]

        plt.plot(e.flatten(), label='%s_%d' % (o, r))


plt.legend()
plt.show()
