import numpy as np
import pandas as pd
from sklearn.svm import SVC


def one_vs_all(X_train, y_train, X_test, C, gamma):
    predictions = {}
    for number in range(10):
        yy = np.where(y_train == number, 1, 0)
        classifier = SVC(C=C, gamma=gamma, kernel='rbf')
        classifier.fit(X_train, yy)
        predictions[number] = classifier.predict(X_test)
    return predictions

