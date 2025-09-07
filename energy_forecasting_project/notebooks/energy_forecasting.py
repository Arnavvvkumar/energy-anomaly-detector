"""
Energy Forecasting Project - Main Analysis Script
Complete pipeline for forecasting household energy consumption using the UCI dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import custom modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import load_data, clean_data, resample_data, create_features, prepare_training_data, prepare_training_data_flat, select_features
from eda import plot_time_series, plot_distributions, plot_correlations, plot_seasonal_patterns
from baseline_models import train_linear_regression, train_decision_tree, train_xgboost, compare_baseline_models
from advanced_models import train_neural_network
from anomaly_detection import compare_anomaly_detection_methods, plot_anomaly_detection, remove_anomalies
from utils import calculate_metrics, plot_predictions, plot_residuals, plot_model_comparison, save_model


def main():
    """
    Main function to run the complete energy forecasting pipeline.
    """
    print("Starting Energy Forecasting Analysis...")
    
    # 1. Data Loading & Preprocessing
    print("\n1. Loading and preprocessing data...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'household_power_consumption.txt')
    
    df = load_data(data_path)
    df_clean = clean_data(df)
    df_resampled = resample_data(df_clean, 'H')
    df_features = create_features(df_resampled)
    
    print(f"Final dataset shape: {df_features.shape}")
    
    # 2. Anomaly Detection
    print("\n2. Performing anomaly detection...")
    
    target_col = 'Global_active_power'
    target_data = df_features[target_col].values
    
    # Compare different anomaly detection methods
    anomaly_results = compare_anomaly_detection_methods(target_data)
    
    # Use Gaussian method results for cleaning (Andrew Ng's approach)
    best_anomaly_method = 'gaussian'
    anomaly_mask = anomaly_results[best_anomaly_method]['anomaly_mask']
    
    # Plot anomaly detection results
    plot_anomaly_detection(target_data, anomaly_results[best_anomaly_method], 
                          f'Anomaly Detection - {best_anomaly_method.title()}')
    
    # Remove anomalies from the dataset
    print(f"\nRemoving anomalies using {best_anomaly_method}...")
    clean_indices = ~anomaly_mask
    df_clean_features = df_features[clean_indices].copy()
    
    print(f"Original dataset: {df_features.shape[0]} samples")
    print(f"Clean dataset: {df_clean_features.shape[0]} samples")
    print(f"Removed: {df_features.shape[0] - df_clean_features.shape[0]} anomalies")
    
    # 3. Exploratory Data Analysis
    print("\n3. Performing exploratory data analysis...")
    
    plot_time_series(df_clean_features, [target_col], 'Global Active Power Over Time (Clean Data)')
    plot_distributions(df_clean_features, [target_col, 'Voltage', 'Global_intensity'])
    plot_correlations(df_clean_features)
    plot_seasonal_patterns(df_clean_features, target_col)
    
    # 4. Prepare Training Data
    print("\n4. Preparing training data...")
    
    from data_preprocessing import split_data
    train_data, val_data, test_data = split_data(df_clean_features, 0.7, 0.15)
    
    # Use flat data for tree-based models (better approach)
    X_train_flat, y_train = prepare_training_data_flat(train_data, target_col)
    X_val_flat, y_val = prepare_training_data_flat(val_data, target_col)
    X_test_flat, y_test = prepare_training_data_flat(test_data, target_col)
    
    # Feature selection to reduce overfitting
    X_train_selected, X_test_selected, feature_selector = select_features(X_train_flat, y_train, X_test_flat, k=50)
    X_val_selected, _, _ = select_features(X_val_flat, y_val, X_val_flat, k=50)
    
    # 5. Baseline Models
    print("\n5. Training baseline models...")
    
    baseline_results = {}
    
    print("\nTraining Linear Regression...")
    lr_results = train_linear_regression(X_train_selected, y_train, X_test_selected, y_test)
    baseline_results['Linear Regression'] = lr_results['metrics']
    
    print("\nTraining Decision Tree...")
    dt_results = train_decision_tree(X_train_selected, y_train, X_test_selected, y_test)
    baseline_results['Decision Tree'] = dt_results['metrics']
    
    print("\nTraining XGBoost...")
    xgb_results = train_xgboost(X_train_selected, y_train, X_test_selected, y_test)
    baseline_results['XGBoost'] = xgb_results['metrics']
    
    # 6. Model Comparison
    print("\n6. Comparing model performance...")
    
    comparison_df = compare_baseline_models({
        'Linear Regression': lr_results,
        'Decision Tree': dt_results,
        'XGBoost': xgb_results
    })
    
    plot_model_comparison(baseline_results, 'mae')
    plot_model_comparison(baseline_results, 'rmse')
    
    # 7. Best Model Analysis
    print("\n7. Analyzing best performing model...")
    
    best_model_name = comparison_df.loc[comparison_df['Test_MAE'].idxmin(), 'Model']
    print(f"\nBest model: {best_model_name}")
    
    if best_model_name == 'Linear Regression':
        best_results = lr_results
    elif best_model_name == 'Decision Tree':
        best_results = dt_results
    else:
        best_results = xgb_results
    
    # Plot predictions for best model
    plot_predictions(y_test, best_results['predictions']['test'], 
                    f'{best_model_name} - Test Predictions')
    plot_residuals(y_test, best_results['predictions']['test'])
    
    # 8. Save Best Model
    print("\n8. Saving best model...")
    os.makedirs('../models', exist_ok=True)
    model_path = f'../models/best_{best_model_name.lower().replace(" ", "_")}.pkl'
    save_model(best_results['model'], model_path)
    
    # 9. Results Summary
    print("\n9. Results Summary:")
    print("=" * 60)
    print(f"Original dataset: {df_features.shape[0]} samples, {df_features.shape[1]} features")
    print(f"Clean dataset: {df_clean_features.shape[0]} samples (after anomaly removal)")
    print(f"Anomalies removed: {df_features.shape[0] - df_clean_features.shape[0]} samples")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Best model: {best_model_name}")
    print(f"Test MAE: {best_results['metrics']['test_mae']:.4f}")
    print(f"Test RMSE: {best_results['metrics']['test_rmse']:.4f}")
    
    print("\nEnergy Forecasting Analysis Complete!")


if __name__ == "__main__":
    main()
