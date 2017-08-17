import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from kmedians import Kmedians
import utils
from kmeans import Kmeans
from quantize_image import QuantImg
# from kmedians import Kmedians
# from quantize_image import ImageQuantizer
from sklearn.cluster import DBSCAN
import linear_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=['1', '1.1', '1.2', '1.3', '1.4', '2', '2.2', '4', '4.1', '4.3'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1':
        X = utils.load_dataset('clusterData')['X']
        # print(X)
        model = Kmeans(k=4)
        model.fit(X)
        utils.plot_2dclustering(X, model.predict(X))

        plt.xlabel("Random clustering k=4")
        fname = os.path.join("..", "figs", "1.png")

        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    if question == '1.1':
        X = utils.load_dataset('clusterData')['X']
        models = []
        errors = np.zeros(50)
        for i in range(0, 50):
            model = Kmeans(k=4)
            model.fit(X)
            models.append(model)
            errors[i] = model.error(X)

        model = models.pop(np.argmin(errors))
        utils.plot_2dclustering(X, model.predict(X))
        plt.xlabel("Best of 50 clusterings k=4")
        fname = os.path.join("..", "figs", "1.1.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)







    if question == '1.2':
        X = utils.load_dataset('clusterData')['X']

        k = np.arange(0,12)
        min_errors = np.zeros(k.size)
        for kk in k:
            models = []
            errors = np.zeros(50)
            for i in range(0, 50):
                model = Kmeans(k=kk+1)
                model.fit(X)
                models.append(model)
                errors[i] = model.error(X)

            min_errors[kk] = (np.min(errors))

        plt.plot(k, min_errors, label="min_errors")
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.legend()
        fname = os.path.join("..", "figs", "1.2.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

















    if question == '1.3':
        # X = utils.load_dataset('clusterData2')['X']
        # models = []
        # errors = np.zeros(50)
        # for i in range(0, 50):
        #     model = Kmeans(k=4)
        #     model.fit(X)
        #     models.append(model)
        #     errors[i] = model.error(X)
        #
        # model = models.pop(np.argmin(errors))
        # utils.plot_2dclustering(X, model.predict(X))
        # plt.xlabel("clusterData2 k=4")
        # fname = os.path.join("..", "figs", "1.3.png")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)
        #

        # ELBOW

        # X = utils.load_dataset('clusterData2')['X']
        #
        # k = np.arange(0, 15)
        # min_errors = np.zeros(k.size)
        # for kk in k:
        #     models = []
        #     errors = np.zeros(50)
        #     for i in range(0, 50):
        #         model = Kmeans(k=kk + 1)
        #         model.fit(X)
        #         models.append(model)
        #         errors[i] = model.error(X)
        #
        #     min_errors[kk] = (np.min(errors))
        #
        # plt.plot(k, min_errors, label="min_errors")
        # plt.xlabel("k")
        # plt.ylabel("Error")
        # plt.legend()
        # fname = os.path.join("..", "figs", "1.3.graph.pdf")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)




        # MEDIANS
        #
        # X = utils.load_dataset('clusterData2')['X']
        # models = []
        # errors = np.zeros(50)
        # for i in range(0, 50):
        #     model = Kmedians(k=4)
        #     model.fit(X)
        #     models.append(model)
        #     errors[i] = model.error(X)
        #
        # model = models.pop(np.argmin(errors))
        # utils.plot_2dclustering(X, model.predict(X))
        # plt.xlabel("clusterData2 k=4")
        # plt.title('K-medians')
        # fname = os.path.join("..", "figs", "1.3.medians.png")
        # plt.savefig(fname)
        # print("\nFigure saved as '%s'" % fname)
        #





        # ELBOW

        X = utils.load_dataset('clusterData2')['X']

        k = np.arange(0, 15)
        min_errors = np.zeros(k.size)
        for kk in k:
            models = []
            errors = np.zeros(50)
            for i in range(0, 50):
                model = Kmedians(k=kk + 1)
                model.fit(X)
                models.append(model)
                errors[i] = model.error(X)

            min_errors[kk] = (np.min(errors))

        plt.plot(k, min_errors, label="min_errors")
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.legend()
        fname = os.path.join("..", "figs", "1.3.median.graph.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    if question == '1.4':
        X = utils.load_dataset('clusterData2')['X']
        
        model = DBSCAN(eps=20, min_samples=3)
        y = model.fit_predict(X)

        utils.plot_2dclustering(X,y)

        fname = os.path.join("..", "figs", "1.4.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)











    if question == '2':
        img = utils.load_dataset('dog')['I']/255
        model = QuantImg(b=6)
        model.quantize(img)
        img = model.dequantize()




        plt.imshow(img)
        plt.title("b = 6")
        fname = os.path.join("..", "figs", "2.6.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)






    elif question == "4":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Plot data
        plt.figure()
        plt.plot(X, y, 'b.', label="Training data")
        plt.title("Training data")

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X, y)
        print(model.w)

        # Draw model prediction
        Xsample = np.linspace(np.min(X), np.max(X), 1000)[:, None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample, yhat, 'g-', label="Least squares fit", linewidth=4)
        plt.legend()
        figname = os.path.join("..", "figs", "least_squares_outliers.pdf")
        print("Saving", figname)
        plt.savefig(figname)








    elif question == "4.1":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Plot data
        plt.figure()
        plt.plot(X, y, 'b.', label="Training data")
        plt.title("Training data")

        # Fit least-squares estimator
        model = linear_model.WeightedLeastSquares()
        z = np.identity(500)
        for i in range(0, 400):
            z[i][i] = 1
        for i in range(400, 500):
            z[i][i] = 0.1
        model.fit(X, y, z)

        # Draw model prediction
        Xsample = np.linspace(np.min(X), np.max(X), 1000)[:, None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample, yhat, 'g-', label="Weighted least squares fit", linewidth=4)
        plt.legend()
        figname = os.path.join("..", "figs", "4.1.pdf")
        print("Saving", figname)
        plt.savefig(figname)







    elif question == "4.3":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Draw model prediction
        Xsample = np.linspace(np.min(X), np.max(X), 1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample, yhat, 'g-', label = "Linear model gradient fit", linewidth=4)
        plt.legend()
        figname = os.path.join("..","figs","4.3.pdf")
        print("Saving", figname)
        plt.savefig(figname)
