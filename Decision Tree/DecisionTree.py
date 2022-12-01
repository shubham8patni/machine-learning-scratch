



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
    def __init__():