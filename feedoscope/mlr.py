import math
import random
import numpy as np
import pandas as pd
import warnings
import logging

# Utilities

def shuffle(list_1, list_2):
    """Shuffle arrays together in sync."""
    assert list_2.shape[0] == list_1.shape[0], "Arrays must have the same length."
    order = np.random.permutation(list_1.shape[0])
    list1_shuffled = list_1[order, :]
    list2_shuffled = list_2[order, :]
    return list1_shuffled, list2_shuffled


# Min/max normalization of data
def normalize_data(data):  # Assumes columns = features and rows = samples
    mean = np.mean(data, axis=0)
    normRange = np.max(data, axis=0) - np.min(data, axis=0)  # np.std(data, axis=0)

    norm = np.true_divide((data - mean), normRange)

    # to handle the situation where the the denominator equals zero after
    # normalization in some columns, convert the resulting NaNs to 0s
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

    return norm, mean, normRange


class ModifiedLogisticRegression:

    def __init__(self, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Learned parameters of the MLR function are initialised to zero values initially
        self.b = 0
        self.c_hat = 0
        self.feature_weights = None

        # We need to save the normalisation data
        self.mean = None
        self.normRange = None

    def get_params(self, deep=True):
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, s: pd.Series):
        """
        X: input data
        s: known labels
        """
        n, m = X.shape

        # Reshape the series into a 2D numpy array while preserving the order of values
        # s = s.to_numpy()[:, np.newaxis]
        if isinstance(s, pd.Series):
            s = s.to_numpy()
        s = np.asarray(s).reshape(-1, 1)

        # Shuffle the values while preserving their relative order
        X, s = shuffle(X, s)

        # Normalize the data and save the mean and range
        X, mean, normRange = normalize_data(X)
        self.mean = mean
        self.normRange = normRange

        # Add a column of ones
        X = np.concatenate((np.ones((n, 1)), X), axis=1)

        # Updated the learned parameters with initial values
        self.feature_weights = np.ones((1, m + 1))
        self.b = 1

        # Train across the number of epochs
        for i in range(self.epochs):
            # Shuffle data for this epoch
            X, s = shuffle(X, s)

            # Cycle through each datasample (need to vectorize!)
            for t in range(n):
                # Calculate partial derivative components
                e_w = np.exp(np.dot(-self.feature_weights, X[t, :].T))
                d1 = (self.b * self.b) + e_w
                d2 = 1 + d1

                if math.isinf(e_w):
                    dw = np.zeros_like(X[t, :])
                else:
                    dw = ((s[t] - 1) / d1 + 1 / d2) * X[t, :] * e_w

                db = ((1 - s[t]) / d1 - 1 / d2) * 2 * self.b

                self.feature_weights = self.feature_weights + self.learning_rate * dw
                self.b = self.b + self.learning_rate * db
            # logging.info(f'Epoch {i}: b={self.b}, c={self.c_hat}, w={self.feature_weights}')

        # Estimate c=p(s=1|y=1) using learned b value
        self.c_hat = np.divide(1, (1 + (self.b**2)))

        self.classes_ = np.unique(s.flatten())

    def predict_proba(self, X):
        n, m = X.shape

        X = np.concatenate((np.ones((n, 1)), X), axis=1)

        mean = np.concatenate(([0], self.mean))
        normRange = np.concatenate(([1], self.normRange))

        normalizedSample = (X - mean) / normRange
        normalizedSample = np.nan_to_num(
            normalizedSample, nan=0.0, posinf=0.0, neginf=0.0
        )

        e_w = np.exp(np.dot(-normalizedSample, np.transpose(self.feature_weights)))
        e_w = np.nan_to_num(e_w, nan=0.0, posinf=0.0, neginf=0.0)

        s_hat = 1.0 / (1 + (self.b**2) + e_w)
        y_hat = s_hat / self.c_hat

        return y_hat

    def predict(self, X):
        predicted_proba = self.predict_proba(X)

        # Convert proba to list of bools and multiply by 1 to turn into int
        preds = (predicted_proba >= 0.5) * 1
        return preds.flatten()

