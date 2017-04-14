from data_handler import DataHandler
import numpy as np
from sklearn.cluster import KMeans

dh = DataHandler()
dh.collect_data()
batch = 100
for a in dh.actions:
    # For ro

    o_means = []
    e_means = []
    for i in range(batch):
        a.split_train_test(0.0)
        a.scale_dataset()

        o_means.append(np.mean(a.X_train))
        e_means.append(np.mean(a.y_train))

    print "Object means:", np.mean(o_means)
    print "Effect means:", np.mean(e_means)

    # End of ro

    # Epsilon

    obj_cluster = KMeans(n_clusters=4)
    obj_cluster.fit(a.X_train)
    obj_distances = []
    for c_center in obj_cluster.cluster_centers_:
        for c_center1 in obj_cluster.cluster_centers_:
            d = np.linalg.norm(c_center-c_center1)
            if d not in obj_distances:
                obj_distances.append(d)
    print "Obj Cluster distances mean:", np.mean(obj_distances)

    effect_cluster = KMeans(n_clusters=8)
    effect_cluster.fit(a.y_train)
    effect_distances = []
    for c_center in effect_cluster.cluster_centers_:
        for c_center1 in effect_cluster.cluster_centers_:
            d = np.linalg.norm(c_center-c_center1)
            if d not in effect_distances:
                effect_distances.append(d)
    print "Effect Cluster distances mean:", np.mean(effect_distances)