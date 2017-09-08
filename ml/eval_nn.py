import keras.backend as K
import matplotlib.pyplot as plt 
from keras import optimizers
from util import *
import numpy as np
def my_mse(y_true, y_pred):
    #return np.abs((y_true-y_pred)/y_true)
    #return K.mean(np.abs((y_true-y_pred)/y_true))
    return K.mean((y_true-y_pred)**2)

if __name__ == '__main__':
    filepaths = getfiles('../data/07-20-Data/normalized/*.csv')
    del filepaths[15]

    #TODO: ASSIGN LOADED MODEL, ORDER AND TARGET
    modelname = 'models/0yaw500'
    ORDER = 0
    TARGET = 2

    # Load json and create model
    json_file = open(modelname+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    from keras.models import model_from_json
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(modelname+'.h5')
    print('Loaded model from disk')

    mse = []
    mae = []
    avg_corr = []
    # Load validation data
    for filepath in filepaths[15:]:
        X_test, y_test = getdata(filepath, ORDER) # Last 4 for test
        y_test = y_test[:, TARGET]
 
        # Evaluate loaded model on test data
        sgd = optimizers.SGD(lr = 0.005, clipnorm = 1.) 
        loaded_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae'])
        score = loaded_model.evaluate(X_test, y_test, verbose=0)
        mse.append(score[0])
        mae.append(score[1])
        #print("mean_squared_error: %.2f%%" % (score[0]*100))
        #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

        y_pred = loaded_model.predict(X_test)
        avg_corr.append( np.corrcoef( y_pred[:,0], y_test)[0,1] )
        #print("avg_corr_coef: %.2f%%" % (avg_corr * 100))
    print("avg. mean squared error: {0:.4f}".format(np.mean(mse)))
    print("avg. mean absolute error: {0:.4f}".format(np.mean(mae)))
    print("avg. correlation coefficient: {0:.4f}".format(np.mean(avg_corr)))

    # Load data to be visualzed
    X_viz, y_viz = getdata(filepaths[18], ORDER)  #testing the 19th for visualization
    y_viz = y_viz[:, TARGET]

    # Visualize prediction
    y_pred = loaded_model.predict(X_viz)
    plt.plot(y_viz,label='ground truth')
    plt.plot(y_pred,label='prediction')
    #plt.legend(loc=4)
    plt.savefig('images/0yaw500test.png')   # TODO: ASSIGN PICTURE NAME
    plt.show()

    # Visualize model
    #from keras.utils import plot_model
    #plot_model(loaded_model, to_file='images/model_struct.png')

    # Write model weights
    #network = loaded_model.get_weights()
    #with open('weights.txt', 'w') as txtfile:
    #    for layer in network:
    #        for weight in layer:
    #            txtfile.write( str(weight) + '\n')
    #        txtfile.write('\n')

