import numpy as np

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold          # Threshold value for the feature
        self.value = value                  # Output value if it's a leaf node
        self.true_branch = true_branch      # Subtree for samples where condition is true
        self.false_branch = false_branch    # Subtree for samples where condition is false

class RegressionTree:
    def __init__(self, max_depth=5, min_samples_split=2, min_impurity_decrease=1e-7, reg_lambda=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease # Corresponds to gamma in XGBoost paper for pre-pruning based on gain
        self.reg_lambda = reg_lambda # L2 regularization term
        self.gamma = gamma # Minimum loss reduction required to make a further partition on a leaf node of the tree (post-pruning like, but used in gain)
        self.root = None

    def _calculate_split_gain(self, G, H, G_left, H_left, G_right, H_right):
        """
        Calculate gain for a split based on gradients (G) and Hessians (H).
        Gain = 0.5 * [ (sum(G_left)^2 / (sum(H_left) + lambda)) + 
                       (sum(G_right)^2 / (sum(H_right) + lambda)) - 
                       (sum(G_total)^2 / (sum(H_total) + lambda)) ] - gamma
        """
        term_left = (np.sum(G_left)**2) / (np.sum(H_left) + self.reg_lambda)
        term_right = (np.sum(G_right)**2) / (np.sum(H_right) + self.reg_lambda)
        term_total = (np.sum(G)**2) / (np.sum(H) + self.reg_lambda)
        
        gain = 0.5 * (term_left + term_right - term_total) - self.gamma
        return gain

    def _calculate_leaf_value(self, G, H):
        """
        Calculate the optimal leaf value.
        output = - sum(gradients_in_leaf) / (sum(hessians_in_leaf) + lambda_reg)
        """
        return -np.sum(G) / (np.sum(H) + self.reg_lambda)

    def _find_best_split(self, X, G, H):
        best_gain = -float('inf')
        best_feature_idx = None
        best_threshold = None
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None, None, -float('inf')

        # Calculate G_total and H_total for the current node
        G_total_sum = np.sum(G)
        H_total_sum = np.sum(H)

        for feature_idx in range(n_features):
            feature_values = np.unique(X[:, feature_idx])
            
            for threshold in feature_values:
                # Split data
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

        if gain <= self.min_impurity_decrease : # If gain is not significant enough
            leaf_value = self._calculate_leaf_value(G, H)
            return TreeNode(value=leaf_value)

        left_indices = X[:, feature_idx] < threshold
        right_indices = X[:, feature_idx] >= threshold
        
        # Ensure splits are not empty, though _find_best_split should handle this
        if not np.any(left_indices) or not np.any(right_indices):
            leaf_value = self._calculate_leaf_value(G, H)
            return TreeNode(value=leaf_value)

        true_branch = self._build_tree(X[left_indices], G[left_indices], H[left_indices], current_depth + 1)
        false_branch = self._build_tree(X[right_indices], G[right_indices], H[right_indices], current_depth + 1)

        return TreeNode(feature_index=feature_idx, threshold=threshold, true_branch=true_branch, false_branch=false_branch)

    def fit(self, X, G, H): # Gradients (G) and Hessians (H)
        self.root = self._build_tree(X, G, H, 0)

    def predict(self, X):
        predictions = np.array([self._traverse_tree(x, self.root) for x in X])
        return predictions

    def _traverse_tree(self, x, node):
        if node.value is not None: # Leaf node
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
        self.min_impurity_decrease = min_impurity_decrease # For tree pruning based on gain
        self.reg_lambda = reg_lambda # L2 regularization on leaf weights
        self.gamma = gamma # Min loss reduction to make a split (used in tree gain calculation)
        self.base_score = base_score # Initial prediction
        self.trees = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _calculate_gradients_hessians(self, y_true, y_pred_proba):
        # For binary classification with logistic loss
        # y_pred_proba is the probability (output of sigmoid)
        # grad = p - y
        # hess = p * (1 - p)
        gradients = y_pred_proba - y_true
        hessians = y_pred_proba * (1 - y_pred_proba)
        return gradients, hessians

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        
        # Initialize predictions (logits)
        current_predictions_logit = np.full(n_samples, np.log(self.base_score / (1 - self.base_score))) # log-odds for base_score

        for i in range(self.n_estimators):
            # Convert current logit predictions to probabilities
            current_predictions_proba = self._sigmoid(current_predictions_logit)
            
            # Calculate gradients and Hessians
            gradients, hessians = self._calculate_gradients_hessians(y, current_predictions_proba)
            
            # Train a new tree
            tree = RegressionTree(max_depth=self.max_depth, 
                                  min_samples_split=self.min_samples_split,
                                  min_impurity_decrease=self.min_impurity_decrease,
                                  reg_lambda=self.reg_lambda,
                                  gamma=self.gamma)
            tree.fit(X, gradients, hessians)
            
            # Get predictions from the new tree (these are the leaf output values)
            update_values = tree.predict(X)
            
            # Update the overall model predictions (logits)
            current_predictions_logit += self.learning_rate * update_values
            
            self.trees.append(tree)
            if (i + 1) % 10 == 0:
                print(f"Booster {i+1}/{self.n_estimators} built.")


    def predict_proba(self, X):
        if not self.trees:
            # Return base score if no trees are trained (e.g., n_estimators = 0)
            return np.full(X.shape[0], self.base_score)

        # Initial logit predictions
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
