import sklearn 
import sklearn.datasets 
import sklearn.linear_model 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import matplotlib.patches as mpatches

def load_dataset():
    np.random.seed(3) 
    return sklearn.datasets.make_circles(400, noise=0.15, factor=0.2)

def plot_decision_boundary(pred_func, X, Y): 
    # create a mesh of possible values for X1 and X2
    x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h)) 
    # predict output for all the values in the mesh
    Y_hat = pred_func(np.c_[X1.ravel(), X2.ravel()]) 
    Y_hat = Y_hat.reshape(X1.shape) 
    Y_hat = Y_hat.round(decimals=0)
    # plot predicted values
    plt.contourf(X1, X2, Y_hat, cmap=plt.cm.Spectral) 
    # plot points in the training dataset
    reds = Y == 0
    blues = Y == 1
    class_1=plt.scatter(X[reds, 0], X[reds, 1], c="red", s=40, edgecolor='k')
    class_2=plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=40, edgecolor='k')
    # add legend 
    plt.legend((class_1, class_2, mpatches.Patch(color="red"), mpatches.Patch(color="blue")),('No Purchase','Purchase',"No Purchase Class predicted region", "Purchase Class predicted region")) 
    plt.xlabel('Age')
    plt.ylabel('Salary')