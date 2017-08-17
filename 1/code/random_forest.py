import numpy as np
from random_tree import RandomTree
import utils


class RandomForest:


    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth



    def fit(self, X, y):


        listOfModels = []
        for i in range(0,self.num_trees):
            model = RandomTree(max_depth=np.inf)

            model.fit(X,y)
            listOfModels.append(model)
        self.LOM = listOfModels
        self.y_length = y.shape[0]


    def predict(self, X):
        for_calc = np.zeros(self.num_trees)
        y_pred = []

        for tree in self.LOM:
            model = tree

            y_pred.append(model.predict(X))

        y_final = np.zeros(self.y_length)



        for i in range(0, self.y_length):
            k=0
            for j in y_pred:

                for_calc[k] = j[i]
                k = k + 1
            y_final[i] = utils.mode(for_calc)

        return y_final