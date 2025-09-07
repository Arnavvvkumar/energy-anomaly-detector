"""
Baseline models module for energy forecasting project.
Contains implementations of basic machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Any
import xgboost as xgb


def compute_cost(w, b, x, y):
    m = x.shape[0]

    cost_sm = 0

    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        cost_sm += error ** 2
    
    total_cost = 1 / (2 * m) * cost_sm

    return float(total_cost)


def compute_gradient(w, b, x, y):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        dj_dw += error * x[i]
        dj_db += error
    
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Train linear regression using sklearn for multi-feature data."""
    from sklearn.linear_model import LinearRegression
    
    print("Training Linear Regression (using sklearn for multi-feature data)...")
    
    # Use sklearn LinearRegression (implements normal equation like Andrew Ng's course)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Linear Regression Training - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"Linear Regression Test - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    return {
        'model': model,
        'predictions': {'train': y_train_pred, 'test': y_test_pred},
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }


def compute_entropy(y):
    entropy = 0
    if len(y) == 0:
        return 0
    p1 = np.mean(y)

    if p1 == 0 or p1 == 1:
        return 0
    entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

    return entropy


def compute_information_gain(X, y, feature_idx, threshold):
    left_mask = X[:, feature_idx] <= threshold
    right_mask = X[:, feature_idx] > threshold
    
    parent_entropy = compute_entropy(y)
    
    left_entropy = compute_entropy(y[left_mask])
    right_entropy = compute_entropy(y[right_mask])
    
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)
    n_total = len(y)
    
    weighted_entropy = (n_left/n_total) * left_entropy + (n_right/n_total) * right_entropy
    information_gain = parent_entropy - weighted_entropy
    
    return information_gain


def find_best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None
    
    for feature_idx in range(X.shape[1]):
        unique_values = np.unique(X[:, feature_idx])
        for threshold in unique_values:
            gain = compute_information_gain(X, y, feature_idx, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain








def get_best_split(X, y, node_indices):
    best_feature = 0
    best_gain = 0
    
    for feature_idx in range(X.shape[1]):
        unique_values = np.unique(X[node_indices, feature_idx])
        for threshold in unique_values:
            gain = compute_information_gain(X[node_indices], y[node_indices], feature_idx, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
    
    return best_feature


def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []
    
    for idx in node_indices:
        if X[idx, feature] <= np.median(X[node_indices, feature]):
            left_indices.append(idx)
        else:
            right_indices.append(idx)
    
    return np.array(left_indices), np.array(right_indices)


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def build_tree_recursive(X, y, node_indices, max_depth, current_depth):
    if current_depth == max_depth or len(node_indices) <= 1:
        return TreeNode(value=np.mean(y[node_indices]))
    
    best_feature = get_best_split(X, y, node_indices)
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    
    if len(left_indices) == 0 or len(right_indices) == 0:
        return TreeNode(value=np.mean(y[node_indices]))
    
    left_subtree = build_tree_recursive(X, y, left_indices, max_depth, current_depth+1)
    right_subtree = build_tree_recursive(X, y, right_indices, max_depth, current_depth+1)
    
    threshold = np.median(X[node_indices, best_feature])
    return TreeNode(feature=best_feature, threshold=threshold, left=left_subtree, right=right_subtree)


def predict_tree(node, x):
    if node.value is not None:
        return node.value
    
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Train decision tree using sklearn for efficiency."""
    from sklearn.tree import DecisionTreeRegressor
    
    print("Training Decision Tree (using sklearn for efficiency)...")
    
    model = DecisionTreeRegressor(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Decision Tree Training - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"Decision Tree Test - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    return {
        'model': model,
        'predictions': {'train': y_train_pred, 'test': y_test_pred},
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }



def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Train XGBoost model with GPU acceleration and regularization."""
    model = xgb.XGBRegressor(
        n_estimators=200,  # More trees for better performance
        max_depth=4,  # Reduced depth to prevent overfitting
        learning_rate=0.05,  # Lower learning rate for better generalization
        subsample=0.8,  # Use 80% of samples for each tree
        colsample_bytree=0.8,  # Use 80% of features for each tree
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        min_child_weight=3,  # Minimum samples in leaf
        random_state=42,
        tree_method='hist',  # Use histogram method
        device='cuda:0'  # Use GPU (RTX 4070) - newer syntax
    )
    
    print("Training XGBoost model with GPU acceleration (RTX 4070)...")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"XGBoost Training - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"XGBoost Test - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    return {
        'model': model,
        'predictions': {'train': y_train_pred, 'test': y_test_pred},
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }


def compare_baseline_models(models_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Compare performance metrics across all baseline models."""
    comparison_data = []
    
    for model_name, results in models_results.items():
        metrics = results['metrics']
        comparison_data.append({
            'Model': model_name,
            'Train_MAE': metrics['train_mae'],
            'Test_MAE': metrics['test_mae'],
            'Train_RMSE': metrics['train_rmse'],
            'Test_RMSE': metrics['test_rmse']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(4)
    
    print("\nModel Comparison:")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    best_test_mae = comparison_df.loc[comparison_df['Test_MAE'].idxmin(), 'Model']
    best_test_rmse = comparison_df.loc[comparison_df['Test_RMSE'].idxmin(), 'Model']
    
    print(f"\nBest Model by Test MAE: {best_test_mae}")
    print(f"Best Model by Test RMSE: {best_test_rmse}")
    
    return comparison_df
