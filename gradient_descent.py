from online_learning import OnlineLearning
from data_handler import DataHandler
from models import OnlineRegression
import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

ol = OnlineLearning()

def only_regression():
    dh = DataHandler()
    dh.collect_data()
    o = OnlineRegression()
    for a in dh.actions:
        for s in a.samples:
            x_s = ol.scale(s.X)
            y_s = ol.scale(s.y)
            o.update(x_s,y_s)
    plt.plot(o.Js, label="j")
    plt.legend()
    plt.show()
#only_regression()

ol.train()
