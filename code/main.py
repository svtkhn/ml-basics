import sklearn
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import utils

from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest
'''from random_forest import RandomForest'''

from knn import KNN, CNN

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier




def column(matrix, i):
    return [row[i] for row in matrix]




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=["1.1", "2", "2.2", "2.3", "2.4", "3", "3.1", "3.2", "4.1", "4.2", "5"])

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":

        dataset = utils.load_dataset("fluTrends")
        print("Minimum of dataset", ":", np.min(dataset[0]))
        print("Maximum of dataset", ":", np.max(dataset[0]))
        print("Mean of dataset", "   :", np.mean(dataset[0]))
        print("Median of dataset", " :", np.median(dataset[0]))
        print("Mode of dataset", "   :", utils.mode(dataset[0]))


    elif question == "2":

        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "2.2":
        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStump()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2.2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "2.3":
        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2.3_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "2.4":

        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            y_fit = model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree took %f seconds" % (time.time()-t))


        # plt.plot(depths, my_tree_errors, label="mine")

        # PLOT MAP
        # utils.plotClassifier(model, X, y)
        # plt.xlabel("Mine")
        # fname = os.path.join("..", "figs", "q2.4mine.pdf")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)


        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))


        # PLOT MAP
        utils.plotClassifier(model, X, y)
        plt.xlabel("Sklearn")
        fname = os.path.join("..", "figs", "q2.4sklearn.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


        # plt.plot(depths, my_tree_errors, label="sklearn")
        # plt.xlabel("Depth of tree")
        # plt.ylabel("Classification error")
        # plt.legend()
        # fname = os.path.join("..", "figs", "q2.4_tree_errors.pdf")
        # plt.savefig(fname)

        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)







        print("Error: %.3f" % error)


    elif question == "3":
        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "3.1":

        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        depths = np.arange(1,16) # depths to try

        tr_error = np.zeros(depths.size)
        te_error = np.zeros(depths.size)

        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error[i] = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error[i] = np.mean(y_pred != y_test)
            print("Tree depth: ", max_depth)
            print("Training error: %.3f" % tr_error[i])
            print("Testing error: %.3f" % te_error[i])

        plt.plot(depths, tr_error, label="training error")
        plt.plot(depths, te_error, label="testing error")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3.1_tree_errors.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



    elif question == "3.2":

        # dataset = utils.load_dataset("citiesSmall")
        #
        # X_init, y_init = dataset["X"], dataset["y"]
        #
        # b1 = X_init.shape[0] / 2
        # b2 = X_init.shape[0]
        #
        # X, y = X_init[0:b1,0:2], y_init[0:b1]
        # X_test, y_test = X_init[b1:b2, 0:2], y_init[b1:b2]
        #
        # depths = np.arange(1,16) # depths to try
        #
        # tr_error = np.zeros(depths.size)
        # te_error = np.zeros(depths.size)
        #
        # for i, max_depth in enumerate(depths):
        #     model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
        #     model.fit(X, y)
        #
        #     y_pred = model.predict(X)
        #     tr_error[i] = np.mean(y_pred != y)
        #
        #     y_pred = model.predict(X_test)
        #     te_error[i] = np.mean(y_pred != y_test)
        #     print("Tree depth: ", max_depth)
        #     print("Training error: %.3f" % tr_error[i])
        #     print("Validation error: %.3f" % te_error[i])
        #
        # plt.plot(depths, tr_error, label="training error")
        # plt.plot(depths, te_error, label="validation error")
        # plt.xlabel("Depth of tree")
        # plt.ylabel("Error")
        # plt.legend()
        # fname = os.path.join("..", "figs", "q3.2.pdf")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)







        # SWITCHED

        dataset = utils.load_dataset("citiesSmall")

        X_init, y_init = dataset["X"], dataset["y"]

        b1 = X_init.shape[0] / 2
        b2 = X_init.shape[0]

        X, y = X_init[b1:b2, 0:2], y_init[b1:b2]
        X_test, y_test = X_init[0:b1, 0:2], y_init[0:b1]

        depths = np.arange(1, 16)  # depths to try

        tr_error = np.zeros(depths.size)
        te_error = np.zeros(depths.size)

        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error[i] = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error[i] = np.mean(y_pred != y_test)
            print("Tree depth: ", max_depth)
            print("Training error: %.3f" % tr_error[i])
            print("Validation error: %.3f" % te_error[i])

        plt.plot(depths, tr_error, label="training error")
        plt.plot(depths, te_error, label="validation error")
        plt.xlabel("SWITCHED Depth of tree")
        plt.ylabel("Error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3.2switched.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    if question == '4.1':

        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]


        rng = range(1,25)

        tr_error = np.zeros(24)
        te_error = np.zeros(24)


        for i in rng:
            model = KNN(k=i)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error[i-1] = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error[i-1] = np.mean(y_pred != y_test)
            print("\nTraining error for", i, ": %.3f" % tr_error[i-1])
            print("Testing error for", i, ": %.3f" % te_error[i-1])

            # PLOT MAP
            # if i==1:
            #     utils.plotClassifier(model, X, y)
            #
            #     fname = os.path.join("..", "figs", "q4.1map.pdf")
            #     plt.savefig(fname)
            #     print("\nFigure saved as '%s'" % fname)


        # PLOT ERROR GRAPH
        # plt.plot(rng, tr_error, label="training error")
        # plt.plot(rng, te_error, label="testing error")
        # plt.xlabel("K neighbors")
        # plt.ylabel("Error")
        # plt.legend()
        # fname = os.path.join("..", "figs", "q4.1.pdf")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)


    if question == '4.2':
        dataset = utils.load_dataset("citiesBig2")
        X, y = dataset["X"], dataset["y"]+1
        X_test, y_test = dataset["Xtest"], dataset["ytest"]+1

        model = CNN(k=1)
        model.fit(X, y)

        start = time.time()
        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)
        end = time.time()
        print("Time passed:", end - start)

        start = time.time()
        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        end = time.time()
        print("Time passed:", end - start)


        print("\nTraining error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)


        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q4.2.2.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)








    if question == '5':
        dataset = utils.load_dataset('vowel')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']


        forest = RandomForest(num_trees=10,max_depth = 100)
        start = time.time()
        forest.fit(X, y)
        end = time.time() -start
        print("RF fit time:", end)
        start = time.time()
        y_pred = forest.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = forest.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        end =  time.time() -start
        print("RF predict time:", end)
        print("RandomForest training error: %.3f" % tr_error)
        print("RandomForest testing error: %.3f" % te_error)


        # =====================================================

        forest = RandomForestClassifier(max_depth=100)
        start = time.time()

        forest.fit(X, y)
        end = time.time() - start
        print("\nRFC fit time:", end)
        start = time.time()

        y_pred = forest.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = forest.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        end =  time.time() - start
        print("RFC predict time:", end)
        print("RandomForestClassifier training error: %.3f" % tr_error)
        print("RandomForestClassifier testing error: %.3f" % te_error)

        # ==================================================
        #
        # forest = DecisionTree(max_depth = 250)
        # forest.fit(X, y)
        #
        # y_pred = forest.predict(X)
        # tr_error = np.mean(y_pred != y)
        #
        # y_pred = forest.predict(X_test)
        # te_error = np.mean(y_pred != y_test)
        #
        # print("DecisionTree training error: %.3f" % tr_error)
        # print("DecisionTree testing error: %.3f" % te_error)

        # ==================================================

        # forest = RandomTree(max_depth = np.inf)
        # forest.fit(X, y)
        #
        # y_pred = forest.predict(X)
        # tr_error = np.mean(y_pred != y)
        #
        # y_pred = forest.predict(X_test)
        # te_error = np.mean(y_pred != y_test)
        #
        # print("RandomTree training error: %.3f" % tr_error)
        # print("RandomTree testing error: %.3f" % te_error)





