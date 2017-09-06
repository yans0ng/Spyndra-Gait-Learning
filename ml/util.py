import re
import glob
import json
import pandas
import numpy as np
import keras.backend as K

def atoi(text):
    return int(text) if text.isdigit() else 0 #text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def getfiles(req):
    filepaths = glob.glob(req)
    filepaths.sort(key=natural_keys)
    return filepaths

def getdata(filepaths, order):
    # Filepaths can be a single string or a list of string
    if isinstance(filepaths, str):
        df = pandas.read_csv(filepaths)
    else:
        df = pandas.DataFrame()
        for i, filepath in enumerate(filepaths):
            df2 = pandas.read_csv(filepath)
            df = df.append(df2)
    dataset = df.values

    # Features
    X = dataset[:,7:]
    for i in xrange(1, order+1): # TODO: CHANGE MODEL ORDER HERE
        # Merge command and measurement from previous time step
        prev_measure = np.vstack((np.zeros((i,14)),dataset[i:,1:]))
        X = np.hstack((X, prev_measure))

    # Output labels
    y = dataset[:,1:7]

    return X, y

def avg_corr_coef( y_true, y_pred):
    ''' Averaged correlation coefficient '''
    return  K.constant(np.corrcoef( y_true, y_pred )[0,1])

def mse( y_true, y_pred ):
    ''' Mean squared error '''
    return K.mean((y_true - y_pred)**2)
