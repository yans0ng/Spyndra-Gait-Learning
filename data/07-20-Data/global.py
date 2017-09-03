import sklearn
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

source = 'global_data.csv'
data = pandas.read_csv( source, skiprows = [18] ) # walk 18 is bad

# Extract output labels
pos_x = data['px'].as_matrix()
pos_y = data['py'].as_matrix()
orient_x = data['ox'].as_matrix()
orient_y = data['oy'].as_matrix()

y_dist = np.sqrt( pos_x**2.0 + pos_y**2.0)
y_dir = np.arctan2( pos_y, pos_x )
y_orient = np.arctan2( orient_y, orient_x )

y = np.vstack( (y_dist, y_dir, y_orient) ).T

# Extract features
features = ['f1', 'f2', 'f3', 'f4', 't1', 't2', 't3', 't4']
X = data[ features ].as_matrix()

# Separate training and testing dataset
X_train = X[:18]
X_test = X[18:]
y_train = y[:18]
y_test = y[18:]

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train )

y_pred = regr.predict(X_test)

print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

def visualize(ax, labels, prefix, color):
    l = 2
    w = 0.1

    for i, label in enumerate(labels):
        print label
        pos_x = label[0]*np.cos(label[1])
        pos_y = label[0]*np.sin(label[1])
        ax.add_patch( patches.Arrow( pos_x, 
                                     pos_y, 
                                     l*np.cos( label[2]), 
                                     l*np.sin( label[2]), 
                                     width = 2, 
                                     facecolor=color))
        ax.text( pos_x, pos_y+1, prefix + str(i) )

fig, ax = plt.subplots(1,1)
visualize(ax, y_test, 'test ', "red")
visualize(ax, y_pred, "pred ", "blue")
ax.autoscale()
plt.show()
