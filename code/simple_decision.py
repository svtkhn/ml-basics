import numpy as np


def predict(self, X):
    y = np.zeros(X.shape[0])

    for i in range(0, X.shape[0]):
        if X[i, 1] < 36:
            if X[i, 0] < -116:
                y[i] = 1
            else:
                y[i] = 2
        else:
            if X[i, 0] < -97:
                y[i] = 2
            else:
                y[i] = 1

    return y