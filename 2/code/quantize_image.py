import numpy as np
import utils
import math
from kmeans import Kmeans
class QuantImg:

    def __init__(self, b):
        self.b = b

    def quantize(self, X):
        N, D, C = X.shape

        X_reshaped = np.reshape(X, (N*D, C))
        print(X_reshaped)


        model = Kmeans(np.power(2, self.b))
        model.fit(X_reshaped)
        model.predict(X_reshaped)
        y = np.reshape(model.predict(X_reshaped), (N, D))
        self.means = model.means
        self.y = y
        self.X = X




    def dequantize(self):
        X = self.X
        y = self.y
        means = self.means
        N, D, C = X.shape
        for row in range(0,N):
            for col in range(0, D):
                X[row][col][0] = means[y[row][col]][0]
                X[row][col][1] = means[y[row][col]][1]
                X[row][col][2] = means[y[row][col]][2]


        return X





