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

# Basic neural network - fully connected
def basic_model(input_num, neurons ):
    # Create model
    model = Sequential()
    model.add(Dense( neurons[0], input_dim=input_num, kernel_initializer='normal', activation='relu'))
    for i in xrange(1, len(neurons) ):
        model.add( Dense(neurons[i], kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # Optimizer
    sgd = optimizers.SGD(lr = 0.001, clipnorm = 1.)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

# define the LSTM model
def LSTM_model():
    # Create model
    model = Sequential()
    model.add(LSTM(8, input_dim=22, input_length=5400,return_sequences=True))
    model.add(LSTM(22, return_sequences=True))
    model.add(LSTM(14, return_sequences=True))
    model.add(LSTM(6, return_sequences=True))
    model.add(LSTM(4, return_sequences=True))
    model.add(LSTM(1, return_sequences=False))

    # Optimizer
    sgd = optimizers.SGD(lr = 0.005, clipnorm = 1.)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


import keras.backend as K
import matplotlib.pyplot as plt
def my_mse(y_true, y_pred):
    return K.mean(np.abs((y_true-y_pred)/y_true))
    #return np.abs((y_true-y_pred)/y_true)


###########################################################
# main
###########################################################
if __name__ == '__main__':

    # TODO: ASSIGN MODEL ORDER AND NUMBER OF NEURONS
    ORDER = 0
    neurons = [8, 8, 4, 2]
    #ORDER = 1
    #neurons = [22, 14, 8, 4, 2]
    ##ORDER = 2
    ##neurons = [36, 30, 22, 8, 4]
    ##ORDER = 3
    ##neurons = [50, 40, 30, 16, 8]
    # TODO: ASSIGN PREDICT TARGET, HERE I USE YAW
    TARGET = 2  
    # TODO: ASSIGN MODEL NAME
    modelname = 'models/0yaw500'

    # Load dataset
    filepaths = getfiles('../data/07-20-Data/normalized/*.csv')
    print 'Glob {} files'.format(len(filepaths))

    # The 15th only have 105 rows
    del filepaths[15]

    # Fix the random seed for consistency
    #seed = 7
    #np.random.seed(seed)

    # Read data
    X_train, y_train = getdata(filepaths[:15], ORDER) #first 14 for training
    y_train = y_train[:, TARGET] 
    print X_train.shape

    # Train the model
    model = basic_model(8+ORDER*14, neurons)
    model.fit( X_train, y_train, epochs=500,
            batch_size=10, validation_split=0.3, shuffle= True)

    # Serialize model to JSON
    model_json = model.to_json()
    with open(modelname+'.json', 'w') as json_file:
        json_file.write(model_json)
    # Store model weights
    model.save_weights(modelname+'.h5')
    print('Saved model to disk')

    # Load json and create model
    json_file = open(modelname+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    from keras.models import model_from_json
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(modelname+'.h5')
    print('Loaded model from disk')

    # Load testing data
    X_test, y_test = getdata(filepaths[15:], ORDER) # Last 4 for test
    y_test = y_test[:, TARGET]
 
    # Evaluate loaded model on test data
    sgd = optimizers.SGD(lr = 0.005, clipnorm = 1.)
    loaded_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=[my_mse])
    score = loaded_model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    # Load data to be visualzed
    X_viz, y_viz = getdata(filepaths[18], ORDER)  #testing the 19th for visualization
    y_viz = y_viz[:, TARGET]

    # Visualize prediction
    y_pred = loaded_model.predict(X_viz)
    plt.plot(y_pred, 'r',label='prediction')
    plt.plot(y_viz, 'b',label='ground truth')
    plt.legend()
    plt.savefig('images/pred.png')
    plt.show()
