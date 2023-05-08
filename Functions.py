import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd


def one_vs_all(X_train, y_train, X_test, C, gamma):
    classifiers = {}
    predictions = {}
    for number in range(10):
        y = np.where(y_train == str(number), 1, 0)
        classifier = SVC(C=C, gamma=gamma, kernel='rbf')
        classifier.fit(X_train, y)
        classifiers[number] = classifier
        predictions[number] = classifiers[number].predict(X_test)
    return predictions
