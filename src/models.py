import itertools

import numpy as np


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.05, epochs=500, l2=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights = None
        self.bias = 0.0

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -35, 35)))

    def fit(self, x, y):
        y = y.astype(float)
        self.weights = np.zeros(x.shape[1], dtype=float)
        self.bias = 0.0
        n = x.shape[0]

        for _ in range(self.epochs):
            proba = self._sigmoid(x @ self.weights + self.bias)
            error = proba - y
            grad_w = (x.T @ error) / n + self.l2 * self.weights
            grad_b = error.mean()
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        return self

    def predict_proba(self, x):
        proba = self._sigmoid(x @ self.weights + self.bias)
        return np.column_stack([1 - proba, proba])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class RidgeRegressionGD:
    def __init__(self, learning_rate=0.03, epochs=600, l2=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights = None
        self.bias = 0.0

    def fit(self, x, y):
        y = y.astype(float)
        self.weights = np.zeros(x.shape[1], dtype=float)
        self.bias = 0.0
        n = x.shape[0]

        for _ in range(self.epochs):
            pred = x @ self.weights + self.bias
            error = pred - y
            grad_w = (x.T @ error) / n + self.l2 * self.weights
            grad_b = error.mean()
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        return self

    def predict(self, x):
        return x @ self.weights + self.bias


def build_model(name, params):
    if name == "logistic_regression":
        return LogisticRegressionGD(**params)
    if name == "ridge_regression":
        return RidgeRegressionGD(**params)
    raise ValueError(f"Unsupported model: {name}")


def param_grid(tuning_config, base_params):
    if not tuning_config.get("enabled", False):
        return [base_params]

    search_keys = [key for key, value in tuning_config.items() if key != "enabled" and isinstance(value, list)]
    if not search_keys:
        return [base_params]

    combinations = []
    for values in itertools.product(*[tuning_config[key] for key in search_keys]):
        params = base_params.copy()
        params.update(dict(zip(search_keys, values)))
        combinations.append(params)
    return combinations
