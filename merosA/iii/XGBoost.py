import numpy as np

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

class RegressionTree:
    def __init__(self, max_depth=5, min_samples_split=2, min_impurity_decrease=1e-7, reg_lambda=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = None

    def _calculate_split_gain(self, G, H, G_left, H_left, G_right, H_right):
        term_left = (np.sum(G_left)**2) / (np.sum(H_left) + self.reg_lambda)
        term_right = (np.sum(G_right)**2) / (np.sum(H_right) + self.reg_lambda)
        term_total = (np.sum(G)**2) / (np.sum(H) + self.reg_lambda)
        gain = 0.5 * (term_left + term_right - term_total) - self.gamma
        return gain

    def _calculate_leaf_value(self, G, H):
        return -np.sum(G) / (np.sum(H) + self.reg_lambda)

    def _find_best_split(self, X, G, H):
        best_gain = -float('inf')
        best_feature_idx = None
        best_threshold = None
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None, -float('inf')

        for feature_idx in range(n_features):
            feature_values = np.unique(X[:, feature_idx])
            for threshold in feature_values:
                left_indices = X[:, feature_idx] < threshold
                right_indices = X[:, feature_idx] >= threshold

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                G_left, H_left = G[left_indices], H[left_indices]
                G_right, H_right = G[right_indices], H[right_indices]

                current_gain = self._calculate_split_gain(G, H, G_left, H_left, G_right, H_right)

                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_gain

    def _build_tree(self, X, G, H, current_depth):
        if current_depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(G, H)
            return TreeNode(value=leaf_value)

        feature_idx, threshold, gain = self._find_best_split(X, G, H)

        if gain <= self.min_impurity_decrease:
            leaf_value = self._calculate_leaf_value(G, H)
            return TreeNode(value=leaf_value)

        left_indices = X[:, feature_idx] < threshold
        right_indices = X[:, feature_idx] >= threshold
        
        if not np.any(left_indices) or not np.any(right_indices):
            leaf_value = self._calculate_leaf_value(G, H)
            return TreeNode(value=leaf_value)

        true_branch = self._build_tree(X[left_indices], G[left_indices], H[left_indices], current_depth + 1)
        false_branch = self._build_tree(X[right_indices], G[right_indices], H[right_indices], current_depth + 1)

        return TreeNode(feature_index=feature_idx, threshold=threshold, true_branch=true_branch, false_branch=false_branch)

    def fit(self, X, G, H):
        self.root = self._build_tree(X, G, H, 0)

    def predict(self, X):
        predictions = np.array([self._traverse_tree(x, self.root) for x in X])
        return predictions

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
            return self._traverse_tree(x, node.true_branch)
        else:
            return self._traverse_tree(x, node.false_branch)

class XGBoostScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_impurity_decrease=1e-7, 
                 reg_lambda=1.0, gamma=0.0, base_score=0.5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.base_score = base_score
        self.trees = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _calculate_gradients_hessians(self, y_true, y_pred_proba):
        gradients = y_pred_proba - y_true
        hessians = y_pred_proba * (1 - y_pred_proba)
        return gradients, hessians

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        current_predictions_logit = np.full(n_samples, np.log(self.base_score / (1 - self.base_score)))

        for i in range(self.n_estimators):
            current_predictions_proba = self._sigmoid(current_predictions_logit)
            gradients, hessians = self._calculate_gradients_hessians(y, current_predictions_proba)
            tree = RegressionTree(max_depth=self.max_depth, 
                                  min_samples_split=self.min_samples_split,
                                  min_impurity_decrease=self.min_impurity_decrease,
                                  reg_lambda=self.reg_lambda,
                                  gamma=self.gamma)
            tree.fit(X, gradients, hessians)
            update_values = tree.predict(X)
            current_predictions_logit += self.learning_rate * update_values
            self.trees.append(tree)
            if (i + 1) % 10 == 0:
                print(f"Booster {i+1}/{self.n_estimators} built.")

    def predict_proba(self, X):
        if not self.trees:
            return np.full(X.shape[0], self.base_score)
        predictions_logit = np.full(X.shape[0], np.log(self.base_score / (1 - self.base_score)))
        for tree in self.trees:
            predictions_logit += self.learning_rate * tree.predict(X)
        return self._sigmoid(predictions_logit)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_impurity_decrease": self.min_impurity_decrease,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "base_score": self.base_score
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

class MultiClassXGBoost:
    def __init__(self, n_classes, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_impurity_decrease=1e-7, 
                 reg_lambda=1.0, gamma=0.0, base_score=0.5):
        self.n_classes = n_classes
        self.classifiers = [XGBoostScratch(n_estimators=n_estimators, learning_rate=learning_rate, 
                                           max_depth=max_depth, min_samples_split=min_samples_split, 
                                           min_impurity_decrease=min_impurity_decrease, 
                                           reg_lambda=reg_lambda, gamma=gamma, base_score=base_score) 
                            for _ in range(n_classes)]

    def fit(self, X, y):
        for i in range(self.n_classes):
            print(f"Training classifier for class {i}")
            y_binary = (y == i).astype(int)
            self.classifiers[i].fit(X, y_binary)

    def predict_proba(self, X):
        probs = np.array([clf.predict_proba(X) for clf in self.classifiers]).T
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)