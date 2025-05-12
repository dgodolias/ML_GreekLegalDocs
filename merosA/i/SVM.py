import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class SVM:
    def __init__(self, C=1.0, max_iter=10000, dual=True, use_tfidf=True, fit_intercept=True, class_weight=None):
        """Initialize SVM classifier for text classification.
        
        Args:
            C (float): Regularization parameter. Default is 1.0.
            max_iter (int): Maximum number of iterations for optimization. Default is 10000.
            dual (bool): Solve the dual or primal optimization problem. Default is True.
            use_tfidf (bool): Whether to use TF-IDF transformation. Default is True.
            fit_intercept (bool): Whether to calculate the intercept. Default is True.
            class_weight (dict or str): Class weights. Default is None.
        """
        self.C = C
        self.max_iter = max_iter
        self.dual = dual
        self.use_tfidf = use_tfidf
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.model = LinearSVC(
            C=self.C, 
            max_iter=self.max_iter, 
            dual=self.dual,
            fit_intercept=self.fit_intercept,
            class_weight=self.class_weight
        )
        self.calibrated_model = None
        self.tfidf_transformer = TfidfTransformer() if use_tfidf else None
        self.iteration_data = []

    def fit(self, X, y, X_val, y_val):
        """Train the SVM model on binary feature vectors.
        
        Args:
            X (ndarray): Training data feature vectors.
            y (ndarray): Training data labels.
            X_val (ndarray): Validation data feature vectors.
            y_val (ndarray): Validation data labels.
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        # Apply TF-IDF transformation if specified
        X_transformed = self._transform_features(X)
        X_val_transformed = self._transform_features(X_val)
        
        # Fit a calibrated model for probability estimates
        self.calibrated_model = CalibratedClassifierCV(self.model, cv=5)
        self.calibrated_model.fit(X_transformed, y)
        
        # Compute metrics after training
        train_acc, train_matrix = self._compute_metrics(X_transformed, y, print_output=False)
        val_acc, val_matrix = self._compute_metrics(X_val_transformed, y_val, print_output=False)
        
        self.iteration_data.append({
            "iteration": 1,  # Only one iteration for SVM
            "train_accuracy": train_acc,
            "train_matrix": train_matrix,
            "val_accuracy": val_acc,
            "val_matrix": val_matrix
        })
        
        return self.iteration_data

    def _transform_features(self, X):
        """Apply TF-IDF transformation to binary feature vectors if specified.
        
        Args:
            X (ndarray): Binary feature vectors.
        
        Returns:
            ndarray: Transformed feature vectors.
        """
        if self.use_tfidf and self.tfidf_transformer is not None:
            # If first time, fit the transformer
            if not hasattr(self.tfidf_transformer, 'idf_'):
                return self.tfidf_transformer.fit_transform(X).toarray()
            else:
                return self.tfidf_transformer.transform(X).toarray()
        return X

    def predict(self, X):
        """Predict class labels for samples in X.
        
        Args:
            X (ndarray): Feature vectors to predict.
        
        Returns:
            ndarray: Predicted class labels.
        """
        X = np.array(X)
        X_transformed = self._transform_features(X)
        return self.calibrated_model.predict(X_transformed)

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.
        
        Args:
            X (ndarray): Feature vectors to predict probabilities for.
        
        Returns:
            ndarray: Class probabilities.
        """
        X = np.array(X)
        X_transformed = self._transform_features(X)
        return self.calibrated_model.predict_proba(X_transformed)

    def _compute_metrics(self, X, y, set_name="TRAIN", print_output=True):
        """Compute and optionally print metrics for the model.
        
        Args:
            X (ndarray): Feature vectors.
            y (ndarray): True labels.
            set_name (str): Name of the dataset (for printing). Default is "TRAIN".
            print_output (bool): Whether to print the metrics. Default is True.
            
        Returns:
            tuple: (accuracy, metrics_matrix).
        """
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
        """Calculate precision, recall, and F1-score for each class and averages.
        
        Args:
            y_true (ndarray): True labels.
            y_pred (ndarray): Predicted labels.
            
        Returns:
            ndarray: Matrix of metrics.
        """
        classes = np.unique(y_true)
        metrics = []

        # Header
        metrics.append(["Category", "Precision", "Recall", "F1-Score"])

        # Per-class metrics
        for cls in classes:
            TP = np.sum((y_true == cls) & (y_pred == cls))
            FP = np.sum((y_true != cls) & (y_pred == cls))
            FN = np.sum((y_true == cls) & (y_pred != cls))
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.append([f"Class {cls}", round(precision, 4), round(recall, 4), round(f1, 4)])

        # Micro-average
        TP_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
        FP_total = np.sum([np.sum((y_true != cls) & (y_pred == cls)) for cls in classes])
        FN_total = np.sum([np.sum((y_true == cls) & (y_pred != cls)) for cls in classes])
        
        micro_prec = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
        micro_rec = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0
        
        metrics.append(["Micro-Average", round(micro_prec, 4), round(micro_rec, 4), round(micro_f1, 4)])

        # Macro-average
        class_metrics = metrics[1:len(classes)+1]
        precisions = [float(row[1]) for row in class_metrics]
        recalls = [float(row[2]) for row in class_metrics]
        f1s = [float(row[3]) for row in class_metrics]
        
        macro_prec = np.mean(precisions)
        macro_rec = np.mean(recalls)
        macro_f1 = np.mean(f1s)
        
        metrics.append(["Macro-Average", round(macro_prec, 4), round(macro_rec, 4), round(macro_f1, 4)])

        return np.array(metrics)


class MultiClassSVM:
    def __init__(self, C=1.0, max_iter=10000, dual=True, use_tfidf=True, fit_intercept=True, class_weight=None):
        """Initialize multi-class SVM classifier.
        
        Args:
            C (float): Regularization parameter. Default is 1.0.
            max_iter (int): Maximum number of iterations for optimization. Default is 10000.
            dual (bool): Solve the dual or primal optimization problem. Default is True.
            use_tfidf (bool): Whether to use TF-IDF transformation. Default is True.
            fit_intercept (bool): Whether to calculate the intercept. Default is True.
            class_weight (dict or str): Class weights. Default is None.
        """
        self.C = C
        self.max_iter = max_iter
        self.dual = dual
        self.use_tfidf = use_tfidf
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.classifiers = []
        
    def fit(self, X, y, X_val, y_val):
        """Train the multi-class SVM model.
        
        Args:
            X (ndarray): Training data feature vectors.
            y (ndarray): Training data labels.
            X_val (ndarray): Validation data feature vectors.
            y_val (ndarray): Validation data labels.
        """
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # For multi-class problems with many classes, one-vs-rest is more efficient
        if n_classes > 2:
            self.model = SVM(
                C=self.C,
                max_iter=self.max_iter,
                dual=self.dual,
                use_tfidf=self.use_tfidf,
                fit_intercept=self.fit_intercept,
                class_weight=self.class_weight
            )
            
            # Print just once before training
            print(f"Training SVM classifier with {n_classes} classes...")
            
            # We only need one model for multi-class classification
            return self.model.fit(X, y, X_val, y_val)
        else:
            # For binary, keep the same structure as other models
            self.classifiers = [SVM(
                C=self.C,
                max_iter=self.max_iter,
                dual=self.dual,
                use_tfidf=self.use_tfidf,
                fit_intercept=self.fit_intercept,
                class_weight=self.class_weight
            )]
            
            print(f"Training SVM classifier for binary classification...")
            return self.classifiers[0].fit(X, y, X_val, y_val)

    def predict(self, X):
        """Predict class labels for samples in X.
        
        Args:
            X (ndarray): Feature vectors to predict.
        
        Returns:
            ndarray: Predicted class labels.
        """
        if hasattr(self, 'model'):
            return self.model.predict(X)
        else:
            return self.classifiers[0].predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.
        
        Args:
            X (ndarray): Feature vectors to predict probabilities for.
        
        Returns:
            ndarray: Class probabilities.
        """
        if hasattr(self, 'model'):
            return self.model.predict_proba(X)
        else:
            return self.classifiers[0].predict_proba(X)
