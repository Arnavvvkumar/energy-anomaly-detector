import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Any

# Enable GPU acceleration for TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU acceleration enabled: {physical_devices[0]}")
else:
    print("No GPU detected, using CPU")




def train_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        hidden_layers: list = [64, 32]) -> Dict[str, Any]:
    """Train Multi-Layer Perceptron (MLP) neural network with GPU acceleration."""

    # Standardize features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build neural network model
    model = Sequential()
    
    # Input layer
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(X_train.shape[1],)))
    
    # Hidden layers
    for layer_size in hidden_layers[1:]:
        model.add(Dense(layer_size, activation='relu'))
    
    # Output layer (single neuron for regression)
    model.add(Dense(1, activation='linear'))
    
    # Compile model with Adam optimizer (as used in Andrew Ng's course)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    
    print("Training Neural Network (MLP) with GPU acceleration...")
    print(f"Model architecture: {hidden_layers} hidden layers")
    
    # Train the model
    history = model.fit(X_train_scaled, y_train, 
                       epochs=100, 
                       batch_size=32, 
                       validation_split=0.2,
                       verbose=0)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled, verbose=0).flatten()
    y_test_pred = model.predict(X_test_scaled, verbose=0).flatten()
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Neural Network Training - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"Neural Network Test - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'predictions': {'train': y_train_pred, 'test': y_test_pred},
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'history': history.history
    }


