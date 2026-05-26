import random
import time

import numpy as np


def c(samples_number):
    return 2.0 * (np.log(samples_number - 1) + 0.5772156649) - 2.0 * (samples_number - 1) / samples_number


class Node:
    def __init__(self, feature=None, threshold=None, left_node=None, right_node=None, samples_number=-1):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.samples_number = samples_number


class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X):
        self.root = self._fit(X, depth=0)

    def _fit(self, X, depth):
        if len(X) <= 1 or (depth >= self.max_depth):
            return Node(samples_number=len(X))

        feature = np.random.randint(0, X.shape[1])
        min_val, max_val = np.min(X[:, feature]), np.max(X[:, feature])
        threshold = np.random.uniform(min_val, max_val)

        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask

        left = self._fit(X[left_mask], depth + 1)
        right = self._fit(X[right_mask], depth + 1)

        return Node(feature, threshold, left, right)

    def path_length(self, X):
        path_lengths = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            node = self.root
            length = 0
            while node.left_node is not None and node.right_node is not None:
                length += 1
                if x[node.feature] < node.threshold:
                    node = node.left_node
                else:
                    node = node.right_node
            path_lengths[i] = length + (0 if node.samples_number < 2 else c(node.samples_number))
        return path_lengths

    def deviation(self, X):
        deviations = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            node = self.root
            length = 0
            deviation = 0.0
            while node.left_node is not None and node.right_node is not None:
                deviation += np.abs(x[node.feature] - node.threshold)
                length += 1
                if x[node.feature] < node.threshold:
                    node = node.left_node
                else:
                    node = node.right_node
            deviations[i] = deviation / length
        return deviations


class IsolationForest:
    def __init__(self, trees_number=100, samples_number=256, seed=None):
        self.trees_number = trees_number
        self.samples_number = samples_number
        self.max_depth = np.ceil(np.log2(samples_number))
        self.trees = []
        self.cpu_stages_fit_time = None
        self.gpu_stages_fit_time = None
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(random.randint(0, 2**32 - 1))

    def fit(self, X):
        cpu_stage_1_start_time = time.perf_counter_ns()

        n_samples = X.shape[0]
        self.trees = []
        for i in range(self.trees_number):
            indices = np.random.choice(n_samples, self.samples_number, replace=True)
            sub_samples = X[indices]
            tree = IsolationTree(self.max_depth)
            tree.fit(sub_samples)
            self.trees.append(tree)

        cpu_stage_1_stop_time = time.perf_counter_ns()
        self.cpu_stages_fit_time = cpu_stage_1_stop_time - cpu_stage_1_start_time
        self.gpu_stages_fit_time = 0.0

    def clear_no_subsample(self):
        self.trees = []
    def fit_no_subsample(self, X):
        tree = IsolationTree(self.max_depth)
        tree.fit(X)
        self.trees.append(tree)

    def decision_function(self, X):
        path_lengths = np.zeros((len(self.trees), X.shape[0]))
        for i, tree in enumerate(self.trees):
            path_lengths[i] = tree.path_length(X)

        h_avg = np.mean(path_lengths, axis=0)
        return 2 ** (-h_avg / c(self.samples_number))

    def new_decision_function(self, X):
        path_lengths = np.zeros((len(self.trees), X.shape[0]))
        for i, tree in enumerate(self.trees):
            path_lengths[i] = tree.path_length(X)

        deviations = np.zeros((len(self.trees), X.shape[0]))
        for i, tree in enumerate(self.trees):
            deviations[i] = tree.deviation(X)

        h_avg = np.mean(path_lengths, axis=0)
        dev_avg = np.mean(deviations, axis=0)
        return 2 ** (-h_avg / c(self.samples_number)) * dev_avg

    def predict(self, X):
        scores = self.decision_function(X)
        threshold = 0.5
        return np.where(scores < threshold, 0, 1)

    def algorithm_name(self):
        return "IsolationForest"
