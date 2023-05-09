import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd


zzz = "prin"
kk = "s"

mnist = fetch_openml('mnist_784', version=1)
#%%
X, y = mnist.data.values, mnist.target.values
random = np.random.randint(1, 1001)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000, random_state=random)
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=2000, random_state=random)
scaler = StandardScaler()
X_train_sample_scaled = scaler.fit_transform(X_train_sample)

def one_vs_all(X_train, y_train, X_test, C, gamma):
    classifiers = {}
    predictions = {}
    for number in range(10):
        yy = np.where(y_train == str(number), 1, 0)
        classifier = SVC(C=C, gamma=gamma, kernel='rbf')
        classifier.fit(X_train, yy)
        classifiers[number] = classifier
        predictions[number] = classifiers[number].predict(X_test)
    return predictions

one_vs_all(X_train_sample_scaled, y_train_sample, scaler.transform(X_test), 4.92, 0.001)