import numpy as np



class Node:
    def __init__(self, features=None, threshold = None, left =None , right = None, *, value = None):
        self.feature = features
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 100, n_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None


    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y):
        n_samples, n_feats = X.shape
        n_labels = np.unique(y)

        # check stopping criteria

        # find best split

        # create child nodes

    
    def predict():