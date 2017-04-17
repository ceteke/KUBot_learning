from online_learning import OnlineLearning
from data_handler import DataHandler
from models import OnlineRegression
import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

def scale(features):
    features = np.delete(features, 3)
    position = minmax_scale(features[0:3])
    position = np.multiply(position, 10.0)
    size = minmax_scale(features[3:6])
    histogram = minmax_scale(features[6:51])
    histogram = np.multiply(histogram, 0.1)
    new_feats = np.append(position, np.append(size, histogram))
    return new_feats

def only_regression():
    dh = DataHandler()
    dh.collect_data()
    o = OnlineRegression()
    for a in dh.actions:
        for s in a.samples:
            x_s = scale(s.X)
            y_s = scale(s.y)
            o.update(x_s,y_s)
    plt.plot(o.Js, label="j")
    plt.legend()
    plt.show()
#only_regression()
ol = OnlineLearning()
ol.train()
