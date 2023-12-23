import numpy as np
from DecisionTrees import DecisionTree
from collections import Counter


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_feature = n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_feature,
            )
            X_sample, y_sample = self.start_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def start_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_pred = np.swapaxes(predictions, 0, 1)
        y_pred = np.array([self._most_common_label(pred) for pred in tree_pred])
        return y_pred
