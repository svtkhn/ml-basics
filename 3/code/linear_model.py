import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils
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

# Least Squares with a bias added
class LeastSquaresBias:
    def __init__(self):
        pass

    def fit(self,X,y):
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        a = np.dot(X.T, X)
        b = np.dot(X.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):
        ones = np.ones((Xhat.shape[0], 1))
        Xhat = np.concatenate((ones, Xhat), axis=1)
        w = self.w
        yhat = np.dot(Xhat, w)
        return yhat

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):

        X_new = np.ones((X.shape[0],1))
        for i in range(1, self.p+1):
            X_new = np.concatenate((X_new, np.power(X, i)), axis=1)
        self.leastSquares.fit(X_new,y)

    def predict(self, Xhat):

        X_new = np.ones((Xhat.shape[0],1))
        for i in range(1, self.p+1):
            X_new = np.concatenate((X_new, np.power(Xhat, i)), axis=1)
        return self.leastSquares.predict(X_new)




    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):

        n = X.shape[0]
        d = self.p + 1
        # Z should have as many rows as X and as many columns as (p+1)
        Z = np.ones((n, d))

        ''' YOUR CODE HERE FOR Q1.2'''



        return Z

# Least Squares with RBF Kernel
class LeastSquaresRBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self,X,y):
        self.X = X
        [n, d] = X.shape
        Z = self.__rbfBasis(X, X, self.sigma)

        # Solve least squares problem
        l = 1e-12

        a = Z.T.dot(Z) + l * np.identity(n)
        b = np.dot(Z.T, y)
        self.w = solve(a,b)

    def predict(self, Xtest):
        Z = self.__rbfBasis(Xtest, self.X, self.sigma)
        yhat = Z.dot(self.w)
        return yhat

    def __rbfBasis(self, X1, X2, sigma):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        den = 1 / np.sqrt(2 * np.pi * (sigma** 2))

        D = (X1**2).dot(np.ones((d, n2))) + \
            (np.ones((n1, d)).dot((X2.T)** 2)) - \
            2 * (X1.dot( X2.T))

        Z = den * np.exp(-1* D / (2 * (sigma**2)))
        print(X1.shape)
        print(X2.shape)

        print(Z.shape)
        return Z


class logReg:
    # Logistic Regression
    def __init__(self, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)


        return np.sign(yhat)

    def decision_function(self, X):
        return np.dot(X, self.w)

class logRegL2:
    # Logistic Regression
    def __init__(self, lammy, verbose=1, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))+(self.lammy/2)*w.T.dot(w)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)+self.lammy*w

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        # utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)


class logRegL1:
    # Logistic Regression
    def __init__(self, lammy, verbose=1, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + (self.lammy) * np.sum(w)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy
        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.lammy,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)
        print(w)
        print(X)
        return np.sign(yhat)


class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, lammy=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # add "i" to the set
                (w, f) = minimize(list(selected_new))
                if f<minLoss:
                    minLoss=f
                    bestFeature=i
                # then compute the score and update the minLoss/bestFeature

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    # a silly classifier that uses least squares
    def __init__(self):
        pass

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((d, self.n_classes))
        print(y.shape)

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            self.W[:, i] = np.linalg.lstsq(np.dot(X.T, X), np.dot(X.T, ytmp))[0]


    def predict(self, X):
        yhat = np.dot(X, self.W)
        print(X.shape)
        print(self.W.shape)
        return np.argmax(yhat, axis=1)


class logLinearClassifier:
    def __init__(self, maxEvals, verbose):
        self.maxEvals = maxEvals
        self.verbose = verbose

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        models = []
        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1
            model = logReg(maxEvals=self.maxEvals, verbose=self.verbose)
            model.fit(X, ytmp)
            models.append(model)


        self.models= models

    def predict(self, X):
        models = self.models
        yhat = np.zeros((X.shape[0], self.n_classes))
        i = 0
        for model in models:
            yhat[:, i] = model.decision_function(X)
            i += 1
        return np.argmax(yhat, axis=1)


class softmaxClassifier:
    def __init__(self, maxEvals=100):
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        n, d = X.shape
        w = w.reshape((d, self.n_classes))

        ytemp = np.zeros((n, self.n_classes))
        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1
            ytemp[:, i] = ytmp

        f = 0
        for i in range(n):
            sub = 0
            for j in range(self.n_classes):
                sub = sub + np.exp(np.dot(w[:, j], X[i]))
            f = f - np.dot(w[:, y[i]].T, X[i]) + np.log(sub)


        g=np.zeros((d, self.n_classes))

        for cc in range(self.n_classes):
            for dd in range(d):
                for i in range(n):
                    sub = 0
                    for j in range(self.n_classes):
                        sub = sub + np.exp(np.dot(w[:, j], X[i]))
                    # for c in range(self.n_classes):
                    g[dd][cc] = g[dd][cc] - X[i][dd]*(ytemp[i][cc] == 1) + (X[i][dd] * np.exp(np.dot(w[:, cc], X[i]))) / sub

        g = g.flatten()
        print(g)
        return f, g

    def fit(self,X, y):
        check = np.arange(0,15)

        n, d = X.shape
        self.n_classes = np.unique(y).size
        self.w = np.zeros((d, self.n_classes))
        self.w = self.w.flatten()
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y)

        self.w = self.w.reshape((d,self.n_classes))

    def predict(self, X):
        W = self.w
        yhat = np.zeros((X.shape[0], self.n_classes))
        for i in range(self.n_classes):
            yhat[:, i] = np.dot(X, W[:,i])
        return np.argmax(yhat, axis=1)