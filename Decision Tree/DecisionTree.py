import numpy as np
from collections  import Counter


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

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = np.unique(y)
        
        # check stopping criteria
        if (depth > self.max_depth or n_labels == 1 or  n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idx = np.random(n_feats. self.n_features, replace = False)

        # find best split
        best_feature, best_thresh =  self._best_split(X, y, feat_idx) # the part where we create randomness in decision trees
        

        # create child nodes


    def _best_split(self, X,y , feat_idxs):
        best_gain = -1 
        split_idx, split_threshold = None, None

        for feat_idx, in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate information gain 
                gain = self._information_gain()

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idx, right_idx = self._split(X_column, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        # calculate the avg. weighted entropy of children

        # calculate the informatiopn gain

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict():