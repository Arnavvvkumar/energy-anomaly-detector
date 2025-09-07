"""
Anomaly Detection module for energy forecasting project.
Contains implementations of various anomaly detection algorithms.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, List


def estimate_gaussian(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate Gaussian parameters (mean and variance) from training data."""

    m, n = X.shape
    
    # Calculate mean for each feature: μᵢ = (1/m) * Σ xᵢ⁽ʲ⁾
    mu = (1/m) * np.sum(X, axis=0)
    
    # Calculate variance for each feature: σᵢ² = (1/m) * Σ (xᵢ⁽ʲ⁾ - μᵢ)²
    sigma2 = (1/m) * np.sum((X - mu) ** 2, axis=0)
    
    return mu, sigma2


def multivariate_gaussian(X: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Compute multivariate Gaussian probability density function."""
    k = len(mu)
    
    # Handle case where sigma2 is a vector (diagonal covariance matrix)
    if sigma2.ndim == 1:
        sigma2 = np.diag(sigma2)
    X_centered = X - mu
    p = (1 / np.sqrt((2 * np.pi) ** k * np.linalg.det(sigma2))) * \
        np.exp(-0.5 * np.sum(X_centered @ np.linalg.inv(sigma2) * X_centered, axis=1))
    
    return p


def select_threshold(y_val: np.ndarray, p_val: np.ndarray) -> Tuple[float, float]:
    """Select optimal threshold using F1-score optimization."""
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    # Try different epsilon values
    step_size = (np.max(p_val) - np.min(p_val)) / 1000
    epsilons = np.arange(np.min(p_val), np.max(p_val), step_size)
    
    for epsilon in epsilons:
        # Classify examples as anomalies if p(x) < epsilon
        predictions = (p_val < epsilon).astype(int)
        
        # Calculate true positives, false positives, false negatives
        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))
        
        # Calculate precision and recall
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        
        # Calculate F1 score
        if precision + recall > 0:
            F1 = 2 * precision * recall / (precision + recall)
        else:
            F1 = 0
        
        # Update best epsilon if F1 score is better
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    
    return best_epsilon, best_F1


def detect_anomalies_gaussian(X_train: np.ndarray, X_val: np.ndarray, 
                             y_val: np.ndarray, epsilon: float = None) -> Dict[str, Any]:
    """Detect anomalies using Gaussian distribution method with F1-score optimization."""
    # Estimate Gaussian parameters from training data
    mu, sigma2 = estimate_gaussian(X_train)
    
    # Calculate probability density for validation set
    p_val = multivariate_gaussian(X_val, mu, sigma2)
    
    # Select best threshold using F1 score if epsilon not provided
    if epsilon is None:
        epsilon, best_F1 = select_threshold(y_val, p_val)
        print(f"Selected epsilon: {epsilon:.6f} with F1 score: {best_F1:.4f}")
    else:
        # Calculate F1 score for given epsilon
        predictions = (p_val < epsilon).astype(int)
        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        
        if precision + recall > 0:
            best_F1 = 2 * precision * recall / (precision + recall)
        else:
            best_F1 = 0
    

    p_train = multivariate_gaussian(X_train, mu, sigma2)
    anomaly_mask = p_train < epsilon
    normal_mask = p_train >= epsilon
    
    results = {
        'method': 'gaussian',
        'mu': mu,
        'sigma2': sigma2,
        'epsilon': epsilon,
        'best_F1': best_F1,
        'anomaly_mask': anomaly_mask,
        'normal_mask': normal_mask,
        'probabilities': p_train,
        'n_anomalies': np.sum(anomaly_mask),
        'n_normal': np.sum(normal_mask)
    }
    
    print(f"Gaussian method detected {results['n_anomalies']} anomalies out of {len(X_train)} samples")
    print(f"Anomaly rate: {results['n_anomalies']/len(X_train)*100:.2f}%")
    print(f"Parameters: μ={mu}, σ²={sigma2}, ε={epsilon:.6f}")
    
    return results


def detect_anomalies_statistical(data: np.ndarray, method: str = 'zscore', threshold: float = 3.0) -> Dict[str, Any]:
    """Detect anomalies using statistical methods (Z-score, IQR, Modified Z-score)."""

    if method == 'zscore':

        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        anomaly_mask = z_scores > threshold
        anomaly_scores = z_scores
        
    elif method == 'iqr':

        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        anomaly_mask = (data < lower_bound) | (data > upper_bound)
        anomaly_scores = np.maximum(data - upper_bound, lower_bound - data) / IQR
        
    elif method == 'modified_zscore':

        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        anomaly_mask = np.abs(modified_z_scores) > threshold
        anomaly_scores = np.abs(modified_z_scores)
    
    else:
        raise ValueError("Method must be 'zscore', 'iqr', or 'modified_zscore'")
    
    normal_mask = ~anomaly_mask
    
    results = {
        'method': method,
        'threshold': threshold,
        'anomaly_mask': anomaly_mask,
        'normal_mask': normal_mask,
        'anomaly_scores': anomaly_scores,
        'n_anomalies': np.sum(anomaly_mask),
        'n_normal': np.sum(normal_mask)
    }
    
    print(f"Statistical ({method}) detected {results['n_anomalies']} anomalies out of {len(data)} samples")
    print(f"Anomaly rate: {results['n_anomalies']/len(data)*100:.2f}%")
    
    return results


def plot_anomaly_detection(data: np.ndarray, results: Dict[str, Any], 
                          title: str = "Anomaly Detection Results") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    axes[0, 0].plot(data, alpha=0.7, label='Normal', color='blue')
    if 'anomaly_mask' in results:
        anomaly_indices = np.where(results['anomaly_mask'])[0]
        axes[0, 0].scatter(anomaly_indices, data[anomaly_indices], 
                          color='red', s=20, label='Anomalies', alpha=0.8)
    axes[0, 0].set_title(f'{title} - Time Series')
    axes[0, 0].set_xlabel('Time Index')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram
    axes[0, 1].hist(data, bins=50, alpha=0.7, color='blue', label='All Data')
    if 'anomaly_mask' in results:
        axes[0, 1].hist(data[results['anomaly_mask']], bins=50, alpha=0.7, 
                       color='red', label='Anomalies')
    axes[0, 1].set_title('Data Distribution')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Anomaly scores
    if 'anomaly_scores' in results:
        axes[1, 0].plot(results['anomaly_scores'], alpha=0.7, color='green')
        if 'threshold' in results:
            axes[1, 0].axhline(y=results['threshold'], color='red', 
                              linestyle='--', label=f'Threshold: {results["threshold"]}')
        axes[1, 0].set_title('Anomaly Scores')
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Anomaly Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    Anomaly Detection Summary:
    
    Total Samples: {len(data)}
    Normal Samples: {results.get('n_normal', 'N/A')}
    Anomaly Samples: {results.get('n_anomalies', 'N/A')}
    Anomaly Rate: {results.get('n_anomalies', 0)/len(data)*100:.2f}%
    
    Method: {results.get('method', 'ML Algorithm')}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                   verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()


def compare_anomaly_detection_methods(data: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Compare different anomaly detection methods and return results."""

    print("Comparing Anomaly Detection Methods (Andrew Ng's Approach)...")
    print("=" * 60)
    
    results = {}
    
    # For single-feature data, reshape to 2D for Gaussian method
    if data.ndim == 1:
        X = data.reshape(-1, 1)
    else:
        X = data
    
    # Create dummy validation data (in real scenario, you'd have labeled validation data)
    # For demonstration, we'll use a simple threshold approach
    print("\n1. Gaussian Distribution Method (with simple threshold):")
    mu, sigma2 = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, sigma2)
    
    # Use a simple threshold (2.5% of data as anomalies)
    epsilon = np.percentile(p, 2.5)
    anomaly_mask = p < epsilon
    normal_mask = p >= epsilon
    
    results['gaussian'] = {
        'method': 'gaussian',
        'mu': mu,
        'sigma2': sigma2,
        'epsilon': epsilon,
        'anomaly_mask': anomaly_mask,
        'normal_mask': normal_mask,
        'probabilities': p,
        'n_anomalies': np.sum(anomaly_mask),
        'n_normal': np.sum(normal_mask)
    }
    
    print(f"Gaussian method detected {results['gaussian']['n_anomalies']} anomalies out of {len(data)} samples")
    print(f"Anomaly rate: {results['gaussian']['n_anomalies']/len(data)*100:.2f}%")
    
    # Statistical methods (also covered in Andrew Ng's course)
    print("\n2. Z-Score Method:")
    results['zscore'] = detect_anomalies_statistical(data, method='zscore', threshold=3.0)
    
    print("\n3. IQR Method:")
    results['iqr'] = detect_anomalies_statistical(data, method='iqr', threshold=1.5)
    
    print("\n4. Modified Z-Score Method:")
    results['modified_zscore'] = detect_anomalies_statistical(data, method='modified_zscore', threshold=3.5)
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION COMPARISON SUMMARY:")
    print("=" * 60)
    
    for method, result in results.items():
        n_anomalies = result.get('n_anomalies', 0)
        anomaly_rate = n_anomalies / len(data) * 100
        print(f"{method.upper():20}: {n_anomalies:4d} anomalies ({anomaly_rate:5.2f}%)")
    
    return results


def remove_anomalies(data: np.ndarray, anomaly_mask: np.ndarray) -> np.ndarray:
    """Remove anomalies from dataset based on anomaly mask."""
    clean_data = data[~anomaly_mask]
    print(f"Removed {np.sum(anomaly_mask)} anomalies. Clean data shape: {clean_data.shape}")
    return clean_data
