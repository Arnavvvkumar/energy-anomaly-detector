"""
Utility functions module for energy forecasting project.
Contains helper functions for evaluation metrics and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Any


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, and R² metrics for model evaluation."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    title: str = "Predictions vs Actual") -> None:
    """Plot actual vs predicted values for model evaluation."""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(f'{title} - Time Series', fontweight='bold')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Scatter Plot', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot residuals for model evaluation.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time Index')
    plt.ylabel('Residuals')
    plt.title('Residuals Over Time', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]], 
                         metric: str = 'mae') -> None:
    """
    Plot comparison of different models.
    
    Args:
        results (Dict[str, Dict[str, float]]): Model results
        metric (str): Metric to compare
    """
    models = list(results.keys())
    train_scores = [results[model].get(f'train_{metric}', 0) for model in models]
    test_scores = [results[model].get(f'test_{metric}', 0) for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
    plt.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel(metric.upper())
    plt.title(f'Model Comparison - {metric.upper()}', fontweight='bold')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        plt.text(i - width/2, train + max(train_scores + test_scores) * 0.01, 
                f'{train:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, test + max(train_scores + test_scores) * 0.01, 
                f'{test:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def save_model(model: Any, filepath: str) -> None:
    """Save trained model to file for future use."""
    import pickle
    import joblib
    
    try:
        if hasattr(model, 'save'):
            model.save(filepath)
        elif hasattr(model, 'save_model'):
            model.save_model(filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        print(f"Model saved to {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")
        try:
            joblib.dump(model, filepath)
            print(f"Model saved using joblib to {filepath}")
        except Exception as e2:
            print(f"Failed to save model: {e2}")


def load_model(filepath: str) -> Any:
    """
    Load trained model from file.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Any: Loaded model object
    """
    import pickle
    import joblib
    
    try:
        if filepath.endswith('.h5') or filepath.endswith('.keras'):
            import tensorflow as tf
            model = tf.keras.models.load_model(filepath)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        else:
            model = joblib.load(filepath)
        
        print(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def create_feature_importance_plot(model: Any, feature_names: List[str]) -> None:
    """
    Create feature importance plot for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (List[str]): Names of features
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance', fontweight='bold')
    
    top_features = min(20, len(feature_names))
    plt.barh(range(top_features), importances[indices[:top_features]])
    plt.yticks(range(top_features), [feature_names[i] for i in indices[:top_features]])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    for i, v in enumerate(importances[indices[:top_features]]):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()


def calculate_confidence_intervals(y_pred: np.ndarray, 
                                 confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for predictions.
    
    Args:
        y_pred (np.ndarray): Predicted values
        confidence_level (float): Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Lower and upper confidence bounds
    """
    from scipy import stats
    
    mean_pred = np.mean(y_pred)
    std_pred = np.std(y_pred)
    n = len(y_pred)
    
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha/2, n-1)
    
    margin_error = t_value * (std_pred / np.sqrt(n))
    
    lower_bound = y_pred - margin_error
    upper_bound = y_pred + margin_error
    
    return lower_bound, upper_bound
