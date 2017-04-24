from data_handler import DataHandler
from sklearn.preprocessing import MinMaxScaler
import pickle

dh = DataHandler()
dh.collect_data()

for a in dh.actions:
    a.split_train_test(0.0)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(a.X_train)
    y_scaler.fit(a.y_train)

    pickle.dump(x_scaler, open('/Users/Cem/learning/models/%s_before_scaler.pkl' % (a.name), 'wb'))
    pickle.dump(y_scaler, open('/Users/Cem/learning/models/%s_effect_scaler.pkl' % (a.name), 'wb'))
