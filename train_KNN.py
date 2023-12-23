import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN

def train_and_predict(X_train, X_test, y_train, y_test):
    clf = KNN(5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = [np.sum(predictions == y_test) * 100 / len(y_test)]
    print("Predictions = ", predictions)
    print("Accuracy = ", (acc))


def default():
    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1234
    )

    plt.figure("KNN")
    plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolors="k", s=20)
    plt.title("Iris Dataset Visualization")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()

    train_and_predict(X_train, X_test, y_train, y_test)


default()
