"""
================
The Iris Dataset
================
This data sets consists of 3 different types of irises'
(Setosa, Versicolour, and Virginica) petal and sepal
length, stored in a 150x4 numpy.ndarray

The rows being the samples and the columns being:
Sepal Length, Sepal Width, Petal Length and Petal Width.

The below plot uses the first two features.
See `here <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ for more
information on this dataset.

"""

# %%
# Importing required packages
# ---------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# %%
# Loading the iris dataset
# ------------------------
from sklearn import datasets


# %
# Placeholder method
def import_data():
    iris = datasets.load_iris()

    # %
    # Dataset information
    print("DATA SET INFO")
    print("Dataset Length: ", len(iris))
#   print("Dataset Shape: ", iris.shape)
#   print("Dataset: ", iris.head())

    return iris


# %%
# Split iris dataset into target variables
# ----------------------------------------
def split_data_set(iris):
    # %
    # Split target variables
    X = iris.data
    y = iris.target

    # %
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X, y, X_train, X_test, y_train, y_test


# %%
# Training algorithms
# -------------------

# %
# Gini
def train_using_gini(X_train, X_test, y_train):
    # %
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=42, max_depth=2, min_samples_leaf=5)

    # %
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# %
# Entropy
def train_using_entropy(X_train, X_test, y_train):
    # %
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=42,
        max_depth=2, min_samples_leaf=5)

    # %
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# %%
# Predictions and placeholders
# ----------------------------

# %
# Function to make predictions
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# %
# Placeholder function for cal_accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))
    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)
    print("Report : ",
          classification_report(y_test, y_pred))


# %%
# Plotting the decision tree
# --------------------------

# %
# Function to plot the decision tree
def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15, 10))
    plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()


if __name__ == "__main__":
    data = import_data()
    X, Y, X_train, X_test, y_train, y_test = split_data_set(data)

    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)

    # %
    # Visualizing the Decision Trees
    plot_decision_tree(clf_gini, ['X1', 'X2', 'X3', 'X4'], ['L', 'B', 'R'])
    plot_decision_tree(clf_entropy, ['X1', 'X2', 'X3', 'X4'], ['L', 'B', 'R'])

# %%
# Operational Phase
# -----------------

# %
# Gini
print("Results Using Gini Index:")
y_pred_gini = prediction(X_test, clf_gini)
cal_accuracy(y_test, y_pred_gini)

# %
# Entropy
print("Results Using Entropy:")
y_pred_entropy = prediction(X_test, clf_entropy)
cal_accuracy(y_test, y_pred_entropy)
