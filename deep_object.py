from data_handler import DataHandler
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D, Conv2D, Dropout, Flatten, TimeDistributed
from keras.losses import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM
import numpy as np
from models import OnlineRegression
import matplotlib.pyplot as plt


def get_nn():
    model = Sequential()
    model.add(TimeDistributed(Dense(128, input_shape=(51,), activation='relu')))
    model.add(LSTM(128))
    model.add(Dense(3, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


dh = DataHandler()
dh.collect_data()

#nn = OnlineRegression(dimensions=(3,52), alpha0=0.2, T=1000)
nn = get_nn()
all_errs = []
for a in dh.actions:
    a.split_train_test(0.05)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(a.X_train)
    y_train = y_scaler.fit_transform(a.y_train)
    y_train_p = []

    X_test = x_scaler.transform(a.X_test)
    y_test = y_scaler.transform(a.y_test)
    y_test_p = []

    for y in y_train:
        y_train_p.append(y[0:3])

    for y in y_test:
        y_test_p.append(y[0:3])

    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train_p[i]
        print x.reshape(1,-1).shape
        nn.fit(x.reshape(1, -1), y.reshape(1, -1), verbose=2, batch_size=1, epochs=10)
        #nn.update(x, y)

    #nn.fit(X_train, y_train_p, batch_size=1, epochs=10)

    for i in range(len(X_test)):
        x = X_test[i]
        y = y_test_p[i]
        y_predicted = nn.predict(x.reshape(1, -1)).flatten()
        #y_predicted = nn.predict(x)
        err = np.linalg.norm(y-y_predicted)
        all_errs.append(err)
        #print err
        #plt.plot(y, label='y')
        #plt.plot(y_predicted, label='y_p')
        #plt.legend()
        #plt.show()
        #all_errs.append(err)

print np.mean(all_errs), len(X_train), len(X_test)