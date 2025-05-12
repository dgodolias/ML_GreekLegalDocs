import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000, tolerance=1e-6, alpha=0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.alpha = alpha
        self.weights = None
        self.bias = None

        
        self.iteration_data = []  

    def fit(self, X_train, y_train, X_val, y_val, batch_size=256):
        X_train = np.array(X_train)
        y_train = np.array(y_train).flatten()
        X_val   = np.array(X_val)
        y_val   = np.array(y_val).flatten()

        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iter):
            dw_sum = 0
            db_sum = 0

       
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                linear_pred = np.dot(X_batch, self.weights) + self.bias
                predictions = sigmoid(linear_pred)

                dw = (1 / len(X_batch)) * np.dot(X_batch.T, (predictions - y_batch))
                db = (1 / len(X_batch)) * np.sum(predictions - y_batch)

                # L1 penalty
                dw += (self.alpha / n_samples) * np.sign(self.weights)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                dw_sum += np.linalg.norm(dw)
                db_sum += abs(db)


            if (i + 1) % 100 == 0:
                print(f"\nIteration {i+1} (max {self.n_iter})")

                train_acc, train_matrix = self._print_iteration_metrics(X_train, y_train, set_name="TRAIN")
 
                val_acc,   val_matrix   = self._print_iteration_metrics(X_val,   y_val,   set_name="VAL")

               
                self.iteration_data.append({
                    "iteration": (i + 1),
                    "train_accuracy": train_acc,
                    "train_matrix": train_matrix,
                    "val_accuracy": val_acc,
                    "val_matrix": val_matrix
                })

            
            if dw_sum < self.tolerance and db_sum < self.tolerance:
                print(f"Convergence reached at iteration {i+1}")
                
                train_acc, train_matrix = self._print_iteration_metrics(X_train, y_train, set_name="TRAIN")
                val_acc,   val_matrix   = self._print_iteration_metrics(X_val,   y_val,   set_name="VAL")
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

    def accuracy(self, y_pred, y_test):
        return np.mean(y_pred == y_test)

    def calculate_metrics(self, y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1_score

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

        matrix = np.array([
            ["Category",      "Precision",       "Recall",         "F1-Score"],
            ["Class 0",       round(precision_0, 4), round(recall_0, 4), round(f1_0, 4)],
            ["Class 1",       round(precision_1, 4), round(recall_1, 4), round(f1_1, 4)],
            ["Micro-Average", round(precision_micro,4), round(recall_micro,4), round(f1_micro,4)],
            ["Macro-Average", round(precision_macro,4), round(recall_macro,4), round(f1_macro,4)],
        ])
        return matrix

    def _print_iteration_metrics(self, X, y, set_name="TRAIN"):
        y_pred = self.predict(X)
        acc = self.accuracy(y_pred, y)
        matrix = self.calculate_metrics_matrix(y, y_pred)
        print(f"==== Metrics on {set_name} set ====")
        print(f"Accuracy = {acc:.4f}")
        print(f"Model Evaluation Metrics ({set_name} Set):")
        print("+" + "-" * 50 + "+")
        for row in matrix:
            print("| {:<12} | {:<12} | {:<12} | {:<12} |".format(*row))
        print("+" + "-" * 50 + "+\n")
        return acc, matrix
