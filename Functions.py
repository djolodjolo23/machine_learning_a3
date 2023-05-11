import numpy as np
from sklearn.svm import SVC


def one_vs_all(X_train, y_train, X_test, C, gamma):
    predictions = {}
    for number in range(10):
        yy = np.where(y_train == number, 1, 0)
        classifier = SVC(C=C, gamma=gamma, kernel='rbf')
        classifier.fit(X_train, yy)
        predictions[number] = classifier.predict(X_test)
    return predictions

def get_nodes_at_some_depth(tree, depth):
    current_depth_nodes = [0]
    for _ in range (depth):
        next_depth_nodes = []
        for node in current_depth_nodes:
            if tree.tree_.children_left[node] != -1:
                next_depth_nodes.append(tree.tree_.children_left[node])
            if tree.tree_.children_right[node] != -1:
                next_depth_nodes.append(tree.tree_.children_right[node])
        current_depth_nodes = next_depth_nodes
    return current_depth_nodes

