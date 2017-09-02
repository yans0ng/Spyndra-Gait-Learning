#!/usr/bin/python
# basic neural network
# [reference]
# http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import numpy as np
import pandas
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from util import *
# load dataset
filepaths = getfiles('data/07-20-Data/normalized/*.csv')
print 'Glob {} files'.format(len(filepaths))

# the 15th only have 105 rows
del filepaths[15]

def getdata(filepaths):
    if isinstance(filepaths, str):
        df = pandas.read_csv(filepaths)
    else:
        df = pandas.DataFrame()
        for i, filepath in enumerate(filepaths):
            df2 = pandas.read_csv(filepath)
            df = df.append(df2)
    dataset = df.values
    X = dataset[:,7:]
    #prev_measure = np.vstack((np.zeros((1,6)),dataset[1:,1:7]))
    #X = np.hstack((X, prev_measure))
    print dataset[1:,:].shape
    prev_measure = np.vstack((np.zeros((1,14)),dataset[1:,1:]))
    X = np.hstack((X, prev_measure))
    y = dataset[:,1:7]
    return X, y

X_train, y_train = getdata(filepaths[:15]) #first 15 for training
y_train = y_train[:,2] #testing yaw
X_test, y_test = getdata(filepaths[15:]) #last 4 for test
y_test = y_test[:,2]
X_viz, y_viz = getdata(filepaths[18])  #testing the 19th for visualization
y_viz = y_viz[:,2]

#TODO test
X_train = X_train.reshape((1,5400,22))
y_train = y_train.reshape((5400,1))
print X_train.shape
print y_train.shape

# fix the random seed for consistency
seed = 7
np.random.seed(seed)
'''
def basic_model():
    # only input and output layer
    model = Sequential()
    model.add(Dense(22, input_dim=22, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=basic_model, nb_epoch=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print "Basic model:"
print("Training Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
model = basic_model()
model.fit(X_train, y_train, epochs = 100, batch_size=5, verbose=0)
model.save_weights("model.h5")
print("Saved model to disk")
'''

# deeper network
# define the model
#def larger_model():
#        # create model
#        model = Sequential()
#        #model.add(Dense(14, input_dim=14, kernel_initializer='normal', activation='relu'))
#        #model.add(Dense(8, kernel_initializer='normal', activation='relu'))
#        #model.add(Dense(4, kernel_initializer='normal', activation='relu'))
#        #model.add(Dense(2, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(1, kernel_initializer='normal'))

#        model.add(Dense(22, input_dim=22, kernel_initializer='normal', activation='relu'))
#        #model.add(Dense(14, kernel_initializer='normal', activation='relu'))
#        model.add(Dense(8, kernel_initializer='normal', activation='relu'))
#        model.add(Dense(4, kernel_initializer='normal', activation='relu'))
#        model.add(Dense(2, kernel_initializer='normal', activation='relu'))
#        model.add(Dense(1, kernel_initializer='normal'))

        # Optimizer
#        sgd = optimizers.SGD(lr = 0.005, clipnorm = 1.)

        # Compile model
#        model.compile(loss='mean_squared_error', optimizer=sgd)
#        return model

# define the model
def larger_model():
        # create model
        model = Sequential()
        #model.add(Dense(14, input_dim=14, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(8, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(4, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(2, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(1, kernel_initializer='normal'))

        model.add(LSTM(22, input_dim=22, input_length=5400,return_sequences=True))
        model.add(LSTM(22, return_sequences=True))
        model.add(LSTM(14, return_sequences=True))
        model.add(LSTM(1, return_sequences=False))
        #model.add(Dense(22, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(14, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(8, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(4, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(2, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(1, kernel_initializer='normal'))

        # Optimizer
        sgd = optimizers.SGD(lr = 0.005, clipnorm = 1.)

        # Compile model
        model.compile(loss='mean_squared_error', optimizer=sgd)
        return model


print "Deeper NN:"
deeper = larger_model()
deeper.fit(X_train,y_train, epochs=500,batch_size=10, validation_split=0.3, shuffle= True)
deeper.save_weights("deep_model.h5")
print("Saved model to disk")

# serialize model to JSON
model_json = deeper.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

import keras.backend as K
import matplotlib.pyplot as plt
def my_mse(y_true, y_pred):
    #return np.abs((y_true-y_pred)/y_true)
    return K.mean(np.abs((y_true-y_pred)/y_true))

from keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("deep_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
sgd = optimizers.SGD(lr = 0.005, clipnorm = 1.)
loaded_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=[my_mse])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


y_pred = loaded_model.predict(X_viz)
plt.plot(y_pred, 'r',label='prediction')
plt.plot(y_viz, 'b',label='ground truth')
plt.legend()
plt.savefig('pred.png')
plt.show()
