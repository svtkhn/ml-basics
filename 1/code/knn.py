"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        distance = np.zeros(self.X.shape[0])
        y_ret = np.zeros(Xtest.shape[0])
        for i in range(Xtest.shape[0]):

            for j in range(self.X.shape[0]):
                distance[j] = np.sqrt((self.X[j][0]-Xtest[i][0])**2+(self.X[j][1]-Xtest[i][1])**2)

            sorted = np.argsort(distance)

            y_ret[i]=utils.mode(self.y[sorted[0:(self.k)]])

            # yhat = utils.mode(self.y[sorted[:min(self.k,len(Xtest))]])

        # print(yhat == self.y)
        return y_ret







class CNN(KNN):

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : an N by D numpy array
        y : an N by 1 numpy array of integers in {1,2,3,...,c}
        """
        print(X.shape)

        Xcondensed = X[0:1,:]
        ycondensed = y[0:1]
        print(Xcondensed.shape)
        print(ycondensed.shape)

        for i in range(1,len(X)):
            x_i = X[i:i+1,:]
            dist2 = utils.euclidean_dist_squared(Xcondensed, x_i)
            inds = np.argsort(dist2[:,0])
            yhat = utils.mode(ycondensed[inds[:min(self.k,len(Xcondensed))]])

            if yhat != y[i]:
                Xcondensed = np.append(Xcondensed, x_i, 0)
                ycondensed = np.append(ycondensed, y[i])

        print(Xcondensed.shape)
        print(ycondensed.shape)

        self.X = Xcondensed
        self.y = ycondensed

