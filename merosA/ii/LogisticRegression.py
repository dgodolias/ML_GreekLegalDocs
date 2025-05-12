import numpy as np

def sigmoid(x):
    """Safe implementation of the sigmoid function to avoid numerical overflow."""
    # Use a threshold to prevent overflow
    clip_value = 700  # Large negative values will be clipped
    x_safe = np.clip(x, -clip_value, clip_value)
    return 1 / (1 + np.exp(-x_safe))

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000, tolerance=1e-6, alpha=0.01, use_stochastic=False):
        self.lr = lr
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.alpha = alpha
        self.use_stochastic = use_stochastic
        self.weights = None
        self.bias = None
        self.iteration_data = []

    def fit(self, X, y, X_val, y_val, batch_size=256):
        X = np.array(X)
        y = np.array(y).flatten()
        X_val = np.array(X_val)
        y_val = np.array(y_val).flatten()

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        if self.use_stochastic:
            # For stochastic training, maintain a moving average of gradient norms for convergence
            grad_norm_window = []
            window_size = 100  # Number of iterations to average for convergence check

            for i in range(self.n_iter):
                # Randomly select a single example
                idx = np.random.randint(0, n_samples)
                X_sample = X[idx:idx+1]
                y_sample = y[idx:idx+1]

                # Compute prediction and gradients
                linear_pred = np.dot(X_sample, self.weights) + self.bias
                prediction = sigmoid(linear_pred)

                dw = np.dot(X_sample.T, (prediction - y_sample))
                db = np.sum(prediction - y_sample)

                # Add L1 regularization term
                dw += self.alpha * np.sign(self.weights)

                # Update weights and bias
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                # Track gradient norm for convergence
                grad_norm = np.linalg.norm(dw) + abs(db)
                grad_norm_window.append(grad_norm)
                if len(grad_norm_window) > window_size:
                    grad_norm_window.pop(0)

                # Store metrics every 100 iterations (but don't print)
                if (i + 1) % 100 == 0:
                    train_acc, train_matrix = self._print_iteration_metrics(X, y, print_output=False)
                    val_acc, val_matrix = self._print_iteration_metrics(X_val, y_val, print_output=False)
                    self.iteration_data.append({
                        "iteration": (i + 1),
                        "train_accuracy": train_acc,
                        "train_matrix": train_matrix,
                        "val_accuracy": val_acc,
                        "val_matrix": val_matrix
                    })

                # Check for convergence based on average gradient norm
                if len(grad_norm_window) == window_size and np.mean(grad_norm_window) < self.tolerance:
                    train_acc, train_matrix = self._print_iteration_metrics(X, y, print_output=False)
                    val_acc, val_matrix = self._print_iteration_metrics(X_val, y_val, print_output=False)
                    self.iteration_data.append({
                        "iteration": (i + 1),
                        "train_accuracy": train_acc,
                        "train_matrix": train_matrix,
                        "val_accuracy": val_acc,
                        "val_matrix": val_matrix
                    })
                    break

        else:
            # Original mini-batch training
            for i in range(self.n_iter):
                dw_sum = 0
                db_sum = 0

                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    X_batch = X[start:end]
                    y_batch = y[start:end]

                    linear_pred = np.dot(X_batch, self.weights) + self.bias
                    predictions = sigmoid(linear_pred)

                    dw = (1 / len(X_batch)) * np.dot(X_batch.T, (predictions - y_batch))
                    db = (1 / len(X_batch)) * np.sum(predictions - y_batch)
                    dw += (self.alpha / n_samples) * np.sign(self.weights)

                    self.weights -= self.lr * dw
                    self.bias -= self.lr * db

                    dw_sum += np.linalg.norm(dw)
                    db_sum += abs(db)

                if (i + 1) % 100 == 0:
                    train_acc, train_matrix = self._print_iteration_metrics(X, y, print_output=False)
                    val_acc, val_matrix = self._print_iteration_metrics(X_val, y_val, print_output=False)
                    self.iteration_data.append({
                        "iteration": (i + 1),
                        "train_accuracy": train_acc,
                        "train_matrix": train_matrix,
                        "val_accuracy": val_acc,
                        "val_matrix": val_matrix
                    })

                if dw_sum < self.tolerance and db_sum < self.tolerance:
                    train_acc, train_matrix = self._print_iteration_metrics(X, y, print_output=False)
                    val_acc, val_matrix = self._print_iteration_metrics(X_val, y_val, print_output=False)
                    self.iteration_data.append({
                        "iteration": (i + 1),
                        "train_accuracy": train_acc,
                        "train_matrix": train_matrix,
                        "val_accuracy": val_acc,
                        "val_matrix": val_matrix
                    })
                    break

        return self.iteration_data

    def predict(self, X):
        X = np.array(X)
        linear_pred = np.dot(X, self.weights) + self.bias
        probs = sigmoid(linear_pred)
        return np.where(probs <= 0.5, 0, 1)

    def predict_proba(self, X):
        X = np.array(X)
        linear_pred = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_pred)

    def _print_iteration_metrics(self, X, y, set_name="TRAIN", print_output=True):
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        matrix = self.calculate_metrics_matrix(y, y_pred)
        if print_output:
            print(f"==== Metrics on {set_name} set ====")
            print(f"Accuracy = {acc:.4f}")
            print(f"Model Evaluation Metrics ({set_name} Set):")
            print("+" + "-" * 50 + "+")
            for row in matrix:
                print("| {:<12} | {:<12} | {:<12} | {:<12} |".format(*row))
            print("+" + "-" * 50 + "+\n")
        return acc, matrix

    def calculate_metrics_matrix(self, y_true, y_pred):
        TP_1 = np.sum((y_true == 1) & (y_pred == 1))
        FP_1 = np.sum((y_true == 0) & (y_pred == 1))
        FN_1 = np.sum((y_true == 1) & (y_pred == 0))
        precision_1 = TP_1 / (TP_1 + FP_1) if (TP_1 + FP_1) > 0 else 0
        recall_1 = TP_1 / (TP_1 + FN_1) if (TP_1 + FN_1) > 0 else 0
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

        TP_0 = np.sum((y_true == 0) & (y_pred == 0))
        FP_0 = np.sum((y_true == 1) & (y_pred == 0))
        FN_0 = np.sum((y_true == 0) & (y_pred == 1))
        precision_0 = TP_0 / (TP_0 + FP_0) if (TP_0 + FP_0) > 0 else 0
        recall_0 = TP_0 / (TP_0 + FN_0) if (TP_0 + FN_0) > 0 else 0
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

        TP = TP_1 + TP_0
        FP = FP_1 + FP_0
        FN = FN_1 + FN_0
        precision_micro = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_micro = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

        precision_macro = (precision_1 + precision_0) / 2
        recall_macro = (recall_1 + recall_0) / 2
        f1_macro = (f1_1 + f1_0) / 2

        return np.array([
            ["Category", "Precision", "Recall", "F1-Score"],
            ["Class 0", round(precision_0, 4), round(recall_0, 4), round(f1_0, 4)],
            ["Class 1", round(precision_1, 4), round(recall_1, 4), round(f1_1, 4)],
            ["Micro-Average", round(precision_micro, 4), round(recall_micro, 4), round(f1_micro, 4)],
            ["Macro-Average", round(precision_macro, 4), round(recall_macro, 4), round(f1_macro, 4)],
        ])

class MultiClassLogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000, tolerance=1e-6, alpha=0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.alpha = alpha
        self.classifiers = []

    def fit(self, X, y, X_val, y_val, batch_size=256):
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        # Use stochastic training if more than 2 classes
        use_stochastic = n_classes > 2
        self.classifiers = [LogisticRegression(
            lr=self.lr,
            n_iter=self.n_iter,
            tolerance=self.tolerance,
            alpha=self.alpha,
            use_stochastic=use_stochastic
        ) for _ in unique_classes]

        for i, cls in enumerate(unique_classes):
            # Print just once per class, not during iterations
            print(f"Training classifier for class {cls} ({i+1}/{len(unique_classes)})")
            y_binary = (y == cls).astype(int)
            y_val_binary = (y_val == cls).astype(int)
            self.classifiers[i].fit(X, y_binary, X_val, y_val_binary, batch_size=batch_size)

        return self.classifiers[0].iteration_data  # Return iteration data from first classifier for compatibility

    def predict(self, X):
        scores = np.array([clf.predict_proba(X) for clf in self.classifiers]).T
        return np.argmax(scores, axis=1)

    def predict_proba(self, X):
        probs = np.array([clf.predict_proba(X) for clf in self.classifiers]).T
        return probs / probs.sum(axis=1, keepdims=True)