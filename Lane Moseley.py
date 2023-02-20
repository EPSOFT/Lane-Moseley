# Author:      Lane Moseley
# Description: This file demonstrates the usage of the k-Nearest Neighbors
#              module implemented in the ML library.
# Resources Used:
#    Iris Dataset:
#         https://archive.ics.uci.edu/ml/datasets/iris
#         https://gist.github.com/curran/a08a1080b88344b0c8a7

import matplotlib.pyplot as plt
from ML import NearestNeighbors, plot_decision_regions
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def main():
    # IRIS DATASET #############################################################
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # Extract the first 100 labels
    y = df.iloc[0:100, 4].values

    # Convert the labels to either 1 or 0
    y = np.where(y == 'Iris-setosa', 0, 1)

    # Extract features from dataset [sepal_length, petal_length]
    X = df.iloc[0:100, [0, 2]].values

    # plot variables
    title = 'Iris Dataset'
    xlabel = 'Sepal Length [cm]'
    ylabel = 'Petal Length [cm]'

    # Plot what we have so far
    # Plot labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the setosa data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # Plot the versicolor data
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # Setup the plot legend
    plt.legend(loc='upper left')

    # Display the plot
    plt.show()

    # scikit-learn k-Nearest Neighbors
    k = 1
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, y, neigh, resolution=0.1, x_label=xlabel, y_label=ylabel,
                          title=title + "\nscikit-learn k-Nearest Neighbors\nk = " + str(k))
    print(title + "\nscikit-learn k-Nearest Neighbors\nk = " + str(k))
    print(classification_report(y, neigh.predict(X)))

    # ML.py k-Nearest Neighbors
    k = 1   # NOTE: the ML.py k-Nearest Neighbors class currently only supports k=1
    knn = NearestNeighbors(k)
    knn.fit(X, y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, y, knn, resolution=0.1, x_label=xlabel, y_label=ylabel,
                          title=title + "\nML.py k-Nearest Neighbors\nk = " + str(k))
    print(title + "\nML.py k-Nearest Neighbors\nk = " + str(k))
    print(classification_report(y, knn.predict(X)))
    ############################################################################

    # IRIS DATASET 2 ###########################################################
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # Extract 100 labels
    y = df.iloc[50:150, 4].values

    # Convert the labels to either 1 or 0
    y = np.where(y == 'Iris-versicolor', 0, 1)

    # Extract features from dataset [sepal_length, petal_length]
    X = df.iloc[50:150, [0, 2]].values

    # plot variables
    title = 'Iris Dataset'
    xlabel = 'Sepal Length [cm]'
    ylabel = 'Petal Length [cm]'

    # Plot what we have so far
    # Plot labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the versicolor data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='versicolor')
    # Plot the virginica data
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='virginica')
    # Setup the plot legend
    plt.legend(loc='upper left')

    # Display the plot
    plt.show()

    # scikit-learn k-Nearest Neighbors
    k = 1
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, y, neigh, resolution=0.1, x_label=xlabel, y_label=ylabel,
                          title=title + "\nscikit-learn k-Nearest Neighbors\nk = " + str(k))
    print(title + "\nscikit-learn k-Nearest Neighbors\nk = " + str(k))
    print(classification_report(y, neigh.predict(X)))

    # ML.py k-Nearest Neighbors
    k = 1   # NOTE: the ML.py k-Nearest Neighbors class currently only supports k=1
    knn = NearestNeighbors(k)
    knn.fit(X, y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, y, knn, resolution=0.1, x_label=xlabel, y_label=ylabel,
                          title=title + "\nML.py k-Nearest Neighbors\nk = " + str(k))
    print(title + "\nML.py k-Nearest Neighbors\nk = " + str(k))
    print(classification_report(y, knn.predict(X)))
    ####################################################################################################################


if __name__ == "__main__":
    main()