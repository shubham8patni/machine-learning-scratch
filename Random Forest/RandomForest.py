from DecisionTree import DecisionTree
import numpy as np

class RandomForest:
    def __init__(self, n_trees = 10, max_depth = 10, min_sample_split = 2, n_feature = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_feature
        self.trees = []

    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_sample_split,n_features=self.n_features)
            X_sample, y_sample = 
            tree.fit()


    def _bootstrap_samples()


    def predict()