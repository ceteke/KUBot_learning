from online_learning import OnlineLearning
from data_handler import DataHandler
from models import OnlineRegression
import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

ol = OnlineLearning()

def only_regression():
    dh = ol.dh
    o = OnlineRegression()
    for a in dh.actions:
        for s in a.samples:
            x_s = ol.scale(s.X)[0:3]
            y_s = np.array(ol.scale(s.y)[0:3])
            o.update(x_s,y_s)
    plt.plot(o.Js, label="j")
    plt.legend()
    plt.show()
#only_regression()

ol.train()
