import numpy as np

class CustomSVC:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.lr * y[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)

class CustomOneVsRestClassifier:
    def __init__(self, estimator, n_classes):
        self.estimator = estimator
        self.n_classes = n_classes
        self.classifiers = [estimator() for _ in range(n_classes)]

    def fit(self, X, y):
        for i in range(self.n_classes):
            y_binary = (y == i).astype(int)
            self.classifiers[i].fit(X, y_binary)

    def predict(self, X):
        decisions = np.array([clf.predict(X) for clf in self.classifiers]).T
        return np.argmax(decisions, axis=1)