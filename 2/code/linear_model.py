import numpy as np
from numpy.linalg import solve
import findMin
import sys
from scipy.optimize import approx_fprime


# Original Least Squares
class LeastSquares:
    # Class constructor
    def __init__(self):
        pass

    def fit(self,X,y):
        # Solve least squares problem

        a = np.dot(X.T, X)
        b = np.dot(X.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):

        w = self.w
        yhat = np.dot(Xhat, w)
        return yhat

# Least Squares where each sample point X has a weight associated with it.
class WeightedLeastSquares:

    def __init__(self):
        pass

    def fit(self,X,y,z):
        print(z.shape)
        X = np.dot(z, X)
        a = np.dot(X.T, X)
        b = np.dot(X.T, y)
        self.w = solve(a, b)

    def predict(self,Xhat):
        w = self.w
        yhat = np.dot(Xhat, w)
        return yhat

class LinearModelGradient:

    def __init__(self):
        pass

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin.findMin(self.funObj, self.w, 100, X, y)

    def predict(self,Xtest):

        w = self.w
        yhat = Xtest*w
        return yhat

    def funObj(self,w,X,y):

        # Calculate the function value
        # f = (1/2)* np.sum((X.dot(w)-y)**2)

        # Calculate the gradient value
        # g = X.T.dot(X.dot(w) - y)


        n, d = X.shape
        f=np.sum(np.log(np.exp(np.dot(X,w)-y)+np.exp(y-np.dot(X,w))))
        g=np.zeros((d,1))
        print(g)
        print(n)
        print(d)
        # for dd in range(0,d):
        for dd in range(0,d):
            for i in range(0, n):
                g[dd][0]= g[dd][0] + (X[i][dd]*np.exp(X[i][dd]*w[dd][0]-y[i][0])-X[i][dd]*np.exp(y[i][0]-X[i][dd]*w[dd][0]))/(np.exp(X[i][dd]*w[dd][0]-y[i][0])+np.exp(y[i][0]-X[i][dd]*w[dd][0]))

        # g = np.reshape(g, (d,1))
        return f, g