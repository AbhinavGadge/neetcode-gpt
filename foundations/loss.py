import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        prob_prod = 0
        epsilon = 1e-7
        for i in range(n):
            y_t, y_p = y_true[i], np.clip(y_pred[i], epsilon, 1 - epsilon)
            prob_prod += y_t * np.log(y_p) + (1-y_t) * np.log(1-y_p)

        return float(round((-1 /n) * prob_prod, 4))

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        prob_prod = 0
        epsilon = 1e-7
        for i in range(n):
            pos = np.where(y_true[i] == 1)
            prob_prod += np.log(y_pred[i][pos] + epsilon)
        return float(round((-1/n) * np.sum(prob_prod), 4))