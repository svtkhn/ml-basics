import numpy as np
import pickle
import sys
import os
import pylab as plt
from sklearn import datasets
from scipy import stats

DATA_DIR = "data"



def load_dataset(dataset_name):
    """Loads the dataset corresponding to the dataset name

    Parameters
    ----------
    dataset_name : name of the dataset

    Returns
    -------
    data :
        Returns the dataset as 'dict'
    """

    if dataset_name == "newsgroups":
        dataset = load_pkl(os.path.join("..","data","newsgroups.pkl"))
        dataset["X"] = dataset["X"].toarray()
        dataset["Xvalidate"] = dataset["Xvalidate"].toarray()

        return dataset

    elif dataset_name == "fluTrends":
        data = load_pkl(os.path.join("..","data","fluTrends.pkl"))

        return data["X"], data["names"]
        
    else:
        return load_pkl(os.path.join('..',DATA_DIR,'{}.pkl'.format(dataset_name)))
        

def plotClassifier(model, X, y):
    """plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    y : it should contain two classes: '1' and '2'

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by D feature array

    y : the N element vector corresponding to the target values

    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line =  np.linspace(x1_min, x1_max, 200)
    x2_line =  np.linspace(x2_min, x2_max, 200)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model.predict(mesh_data)
    y_pred = np.reshape(y_pred, x1_mesh.shape)

    plt.xlim([x1_mesh.min(), x1_mesh.max()])
    plt.ylim([x2_mesh.min(), x2_mesh.max()])

    plt.contourf(x1_mesh, x2_mesh, -y_pred,
                cmap=plt.cm.RdBu, label="decision boundary",
                alpha=0.6)

    plt.scatter(x1[y==1], x2[y==1], color="b", label="class 1")
    plt.scatter(x1[y==2], x2[y==2], color="r", label="class 2")
    plt.legend()
    plt.title("Model outputs '1' for blue region\n"
              "Model outputs '2' for red region")


def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if len(y)==0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]

def load_pkl(fname):
    """Reads a pkl file.

    Parameters
    ----------
    fname : the name of the .pkl file

    Returns
    -------
    data :
        Returns the .pkl file as a 'dict'
    """
    if sys.version_info[0] < 3:
        # Python 2
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        # Python 3
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    return data

def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """

    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)

    # without broadcasting: 
    # n,d = X.shape
    # t,d = Xtest.shape
    # D = X**2@np.ones((d,t)) + np.ones((n,d))@(Xtest.T)**2 - 2*X@Xtest.T
