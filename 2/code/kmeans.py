import numpy as np
import utils
import math
class Kmeans:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]

        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = utils.euclidean_dist_squared(X, means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                means[kk] = X[y==kk].mean(axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break
        self.means = means

    def predict(self, X):
        means = self.means
        dist2 = utils.euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X):
        means = self.means
        dist2 = utils.euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        y = np.argmin(dist2, axis=1)
        errors = np.zeros(X.shape[0])
        for i in range(0, X.shape[0]):
            errors[i] = math.pow(math.hypot(means[y[i]][0] - X[i][0], means[y[i]][1] - X[i][1]), 2)

        return np.sum(errors)
