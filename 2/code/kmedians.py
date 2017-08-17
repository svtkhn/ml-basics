import numpy as np
import utils
import math
class Kmedians:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        medians = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            medians[kk] = X[i]

        while True:
            y_old = y

            # Compute euclidean distance to each median
            dist2 = utils.euclidean_dist_squared(X, medians)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update medians
            for kk in range(self.k):
                medians[kk] = np.median(X[y==kk], axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-medians, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.medians = medians

    def predict(self, X):
        medians = self.medians
        dist2 = utils.euclidean_dist_squared(X, medians)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X):
        medians = self.medians
        dist2 = utils.euclidean_dist_squared(X, medians)
        dist2[np.isnan(dist2)] = np.inf
        y = np.argmin(dist2, axis=1)
        errors = np.zeros(X.shape[0])
        for i in range(0, X.shape[0]):
            errors[i] = abs(math.hypot(medians[y[i]][0] - X[i][0], medians[y[i]][1] - X[i][1]))

        return np.sum(errors)
