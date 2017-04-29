import pickle
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import os
from keras.models import load_model

run_ids = [i for i in range(653, 660)]
objs = { 'box': [ '1318',
                '776',
                '281',
                '592',
                '352',
                '40',
                '535',
                '734',
                '407',
                '595',
                '823',
                '401',
                '41',
                '294',
                '734',
                '712',
                '615',
                '1449',
                '26',
                '141',
                '1448',
                '139',
                '956'],
       'hcylinder': ['843', '1014'],
       'vcylinder': [ '342',
                      '716',
                      '879',
                      '525H',
                      '362',
                      '279',
                      '244',
                      '368*',
                      '536',
                      '508',
                      '341',
                      '877',
                      '1033',
                      '945',
                      '532',
                      '346',
                      '189',
                      '368',
                      '163']}
x_scaler = pickle.load(open('/Volumes/ROSDATA/models/object_scaler.pkl', 'rb'))
y_scaler = pickle.load(open('/Volumes/ROSDATA/models/effect_scaler.pkl', 'rb'))
som = pickle.load(open('/Volumes/ROSDATA/models/som.pkl', 'rb'))
nn = load_model('/Volumes/ROSDATA/models/nn.h5')
for o, rid in objs.iteritems():
    for r in rid:
        for k in run_ids:
            before_csv = '/Volumes/ROSDATA/ros_data/features/new6/%d/push/%s/%s/0.csv' % (k, o, r)
            after_csv = '/Volumes/ROSDATA/ros_data/features/new6/%d/push/%s/%s/1.csv' % (k, o, r)
            if os.path.exists(before_csv):
                before_features = np.genfromtxt(before_csv, delimiter=',')
                after_features = np.genfromtxt(after_csv, delimiter=',')
                break
        if np.array_equal(after_features, np.array([0.0] * 51)):
            print '%s_%s dropped' % (o, r)
            effect_features = deepcopy(before_features)
        else:
            effect_features = np.absolute(np.subtract(after_features, before_features))

        b_a = x_scaler.transform(before_features.reshape(1, -1)).flatten()
        e_a = y_scaler.transform(effect_features.reshape(1, -1)).flatten()[0:3]

        e_p = nn.predict(b_a.reshape(1,-1)).flatten()

        plt.plot([1,2,3], e_a, "o")
        plt.xlim([0.5,3.5])
        plt.xticks([0.5,1,2,3,3.5])


plt.legend()
plt.show()
