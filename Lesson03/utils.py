import sklearn 
import sklearn.datasets 
import sklearn.linear_model 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib

def load_dataset():
    np.random.seed(3) 
    return sklearn.datasets.make_circles(400, noise=0.15, factor=0.2)

def plot_decision_boundary(pred_func, X, Y): 
    # create a mesh of possible values for X1 and X2
    x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01)) 
    # predict output for all the values in the mesh
    Y_hat = pred_func(np.c_[X1.ravel(), X2.ravel()]) 
    Y_hat = Y_hat.reshape(X1.shape) 
    # convert class probabilities to 0 and 1 labels
    Y_hat = Y_hat.round(decimals=0)
    # plot predicted values
    plt.contourf(X1, X2, Y_hat, cmap=plt.cm.Spectral) 
    # plot points in the training dataset
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)