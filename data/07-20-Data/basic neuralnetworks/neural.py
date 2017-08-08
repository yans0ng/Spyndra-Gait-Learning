#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 20:14:47 2017

@author: chen
"""
import csv
import copy
import numpy 
from sklearn.neural_network import MLPRegressor
with open("gait.csv", 'rU') as f:
    reader = csv.reader(f)
    gait = []
    for lines in reader:
        a = []
        for value in lines:
            a.append(float(value))
        #print a
        gait.append(a)
    #print gait
with open("result.csv", 'rU') as f:
    reader = csv.reader(f)
    x = []
    y = []
    oritation = []
    for lines in reader:
        x.append(float(lines[0]))
        y.append(float(lines[1]))
        oritation.append(float(lines[2]))
        #print a
    #print x
    #print y
    #print oritation
a = [0,1,2,3,4]
b = [5,6,7,8,9]
c = [10,11,12,13,14]
d = [15,16,17,18,19]
e = [20,21,22]
data = a+b+c+d
#print data
crossvalidation = [a,b,c,d,e]
#print crossvalidation
def basic_neural_x(crossvalidation,gait,x):
    fx = []
    real = []
    #training for x
    for i in range(len(crossvalidation)-1):
        #print i
        test = crossvalidation[i]
        #print test
        train_temp = copy.deepcopy(crossvalidation)
        train_temp.remove(test)
        trainx = []
        for items in train_temp:
            trainx = trainx+items
        #print trxain
        
    
        train_x=[gait[j] for j in trainx]
        train_yx = [x[j] for j in trainx]
        test_x = [gait[j] for j in test]
        test_yx = [x[j] for j in test]
        #print train_x
        
        #create neural net regressor
        reg = MLPRegressor(hidden_layer_sizes=(8,6,2,1))
        reg.fit(train_x,train_yx)
        # 
        ##test prediction
        # 
        predict=reg.predict(test_x)
        
        #print "_Input_\t_output_"
        for i in range(len(test_x)):
            fx.append(predict[i])
            real.append(test_yx[i])
            #print "  ",test_x[i],"---->",predict[i],test_yx[i]
    print fx,real
    print numpy.corrcoef(fx,real) 

def basic_neural_y(crossvalidation,gait,y):
    fx = []
    real = [] 
    # training for y
    for i in range(len(crossvalidation)-1):
        #print i
        test = crossvalidation[i]
        #print test
        #print test
        train_temp = copy.deepcopy(crossvalidation)
        train_temp.remove(test)
        trainy = []
        for items in train_temp:
            trainy = trainy+items
        train_x=[gait[j] for j in trainy]
        train_yy = [y[j] for j in trainy]
        test_x = [gait[j] for j in test]
        test_yy = [y[j] for j in test]
        #print train_x
        
        #create neural net regressor
        reg = MLPRegressor(hidden_layer_sizes=(10))
        reg.fit(train_x,train_yy)
        # 
        ##test prediction
        # 
        predict=reg.predict(test_x)
        
        #print "_Input_\t_output_"
        for i in range(len(test_x)):
            fx.append(predict[i])
            real.append(test_yy[i])
            #print "  ",test_x[i],"---->",predict[i],test_yx[i]
    print fx,real
    print numpy.corrcoef(fx,real) 

def basic_neural_oritation(crossvalidation,gait,oritation):
    fx = []
    real = []
    #train orit
    for i in range(len(crossvalidation)-1):
        #print i
        test = crossvalidation[i]
        #print test
        train_temp = copy.deepcopy(crossvalidation)
        train_temp.remove(test)
        trainoritation = []
        for items in train_temp:
            trainoritation = trainoritation+items
        #print train
        
    
        train_x=[gait[j] for j in trainoritation]
        train_yoritation = [oritation[j] for j in trainoritation]
        test_x = [gait[j] for j in test]
        test_yoritation = [oritation[j] for j in test]
        #print train_x
        
        #create neural net regressor
        reg = MLPRegressor(hidden_layer_sizes=(8))
        reg.fit(train_x,train_yoritation)
        # 
        ##test prediction
        # 
        predict=reg.predict(test_x)
        
        #print "_Input_\t_output_"
        for i in range(len(test_x)):
            fx.append(predict[i])
            real.append(test_yoritation[i])
            #print "  ",test_x[i],"---->",predict[i],test_yx[i]
    print fx,real
    print numpy.corrcoef(fx,real) 

basic_neural_x(crossvalidation,gait,x)
#basic_neural_y(crossvalidation,gait,y)
#basic_neural_oritation(crossvalidation,gait,oritation)




