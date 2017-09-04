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

# Shuffle samples
shuffle = np.random.choice( len(y), len(y), replace=False )

# Separate training and testing dataset
X_train = X[shuffle[:18]]
X_test = X[shuffle[18:]]
y_train = y[shuffle[:18]]
y_test = y[shuffle[18:]]

# Define Error Metric
def accuracy( y_true, y_pred):
    # Cosine rule
    xt = y_true[:,0] * np.cos( y_true[:,1] )
    yt = y_true[:,0] * np.sin( y_true[:,1] )
    xp = y_pred[:,0] * np.cos( y_pred[:,1] )
    yp = y_pred[:,0] * np.sin( y_pred[:,1] )
    diff = np.sqrt( (xt-xp)**2 + (yt-yp)**2 ) # / y_true[:,0]
    return np.mean(diff)

def score( y_true, y_pred):
    print("\t - Distance MSE: %.2f"% mean_squared_error(y_true[:,0], y_pred[:,0]))
    print("\t - Direction MSE: %.2f"% mean_squared_error(y_true[:,1], y_pred[:,1]))
    print("\t - Orientation MSE: %.6f"% mean_squared_error(y_true[:,2], y_pred[:,2]))
    print("\t - Accuracy: %2f"%accuracy(y_true, y_pred))

# Linear regressor
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train )

y_valid_lr = regr.predict(X_train)  # Training error
y_pred_lr = regr.predict(X_test)    # Testing error

print("Linear Regression")
print("\tTraining")
score(y_train, y_valid_lr)
print("\tTesting")
score(y_test, y_pred_lr)

# Ridge regression
ridge = linear_model.Ridge(alpha=0.5)
ridge.fit(X_train, y_train)

y_valid_rr = ridge.predict(X_train)
y_pred_rr = ridge.predict(X_test)

print("Ridge Regression")
print("\tTraining")
score(y_train, y_valid_rr)
print("\tTesting")
score(y_test, y_pred_rr)

def visualize(ax, labels, colorname):
    l = 4   # arrow length
    w = 0.1 # arrow width
    # Plot arrows
    for i, label in enumerate(labels):
        pos_x = label[0]*np.cos(label[1])
        pos_y = label[0]*np.sin(label[1])
        ax.add_patch( patches.Arrow( pos_x, 
                                     pos_y, 
                                     l*np.cos( label[2]), 
                                     l*np.sin( label[2]), 
                                     width = 2, 
                                     facecolor=colorname))
        ax.text( pos_x-1, pos_y+1, str(i), color=colorname)

# Visualize prediction
fig, ax = plt.subplots(1,1)
visualize(ax, y_test, "red")
visualize(ax, y_pred_lr, "blue")
visualize(ax, y_pred_rr, "green")
ax.grid()
ax.set_xlabel('cm')
ax.set_ylabel('cm')
ax.axis('equal')
ax.autoscale()
plt.show()
