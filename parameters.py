from data_handler import DataHandler
from online_learning import OnlineLearning
import numpy as np
from sklearn.cluster import KMeans

ol = OnlineLearning()
dh = ol.dh
batch = 1
X_train = []
y_train = []

for a in dh.actions:
    # For ro
    for i in range(batch):
        a.split_train_test(0.0)
        for k in range(len(a.train_samples)):
            X_train.append(ol.scale(a.train_samples[k].X))
            y_train.append(ol.scale(a.train_samples[k].y))

    print "Object means:", np.mean(X_train)
    print "Effect means:", np.mean(y_train)

    # End of ro

    # Epsilon
    o_cluster_distances = {}
    obj_cluster = KMeans(n_clusters=12)
    obj_distances = obj_cluster.fit_transform(X_train)
    for o in obj_distances:
        cid = np.argmin(o)
        if cid not in o_cluster_distances:
            o_cluster_distances[cid] = [o[cid]]
        else:
            o_cluster_distances[cid].append(o[cid])
    o_distances = []
    print "Object mean cluster distances:"
    for k, v in o_cluster_distances.iteritems():
        print k, np.mean(v)
        o_distances.append(np.mean(v))
    print np.mean(o_distances)

    e_distances = []
    e_cluster_distances = {}
    efc_cluster = KMeans(n_clusters=28)
    efc_distances = efc_cluster.fit_transform(y_train)
    for e in efc_distances:
        cid = np.argmin(e)
        if cid not in e_cluster_distances:
            e_cluster_distances[cid] = [e[cid]]
        else:
            e_cluster_distances[cid].append(e[cid])
    print "Effect mean cluster distances:"
    for k, v in e_cluster_distances.iteritems():
        print k, np.mean(v)
        e_distances.append(np.mean(v))
    print np.mean(e_distances)
