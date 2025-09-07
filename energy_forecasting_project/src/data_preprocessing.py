import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import Optional


def load_data(file_path: str) -> pd.DataFrame:
    """Load household power consumption data from CSV file and set datetime index."""
    column_names = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    df = pd.read_csv(file_path, sep=';', names=column_names, header=0, na_values=['?'], low_memory=False)
    
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('DateTime', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove missing values and clean the dataset."""
    df_clean = df.copy()
    
    df_clean = df_clean.dropna(how='all')

    df_clean = df_clean.fillna(method='ffill')
    
    df_clean = df_clean.fillna(method='bfill')
    
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    for col in numeric_columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"Data cleaned: {df_clean.shape[0]} rows remaining")
    print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def resample_data(df: pd.DataFrame, frequency: str = 'H') -> pd.DataFrame:
    """Resample data to specified frequency (hourly by default)."""
    df_resampled = df.resample(frequency).mean()
    
    df_resampled = df_resampled.dropna()
    
    print(f"Data resampled to {frequency}: {df_resampled.shape[0]} rows")
    print(f"New date range: {df_resampled.index.min()} to {df_resampled.index.max()}")
    
    return df_resampled


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive feature set with lag, rolling, and interaction features."""
    df_features = df.copy()
    
    # Time-based features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_month'] = df_features.index.day
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    df_features['year'] = df_features.index.year
    
    # Cyclical encoding for time features
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Binary features
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    df_features['is_peak_hour'] = ((df_features['hour'] >= 6) & (df_features['hour'] <= 9) | 
                                  (df_features['hour'] >= 18) & (df_features['hour'] <= 21)).astype(int)
    df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 5)).astype(int)
    df_features['is_workday'] = ((df_features['day_of_week'] >= 0) & (df_features['day_of_week'] <= 4)).astype(int)
    
    # Enhanced lag features (more comprehensive)
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:  # 1h, 2h, 3h, 6h, 12h, 1d, 2d, 3d, 1w
        df_features[f'global_active_power_lag{lag}'] = df_features['Global_active_power'].shift(lag)
        df_features[f'global_reactive_power_lag{lag}'] = df_features['Global_reactive_power'].shift(lag)
        df_features[f'voltage_lag{lag}'] = df_features['Voltage'].shift(lag)
    
    # Rolling statistics (multiple windows)
    for window in [3, 6, 12, 24, 48, 168]:  # 3h, 6h, 12h, 1d, 2d, 1w
        df_features[f'global_active_power_rolling_mean_{window}h'] = df_features['Global_active_power'].rolling(window=window).mean()
        df_features[f'global_active_power_rolling_std_{window}h'] = df_features['Global_active_power'].rolling(window=window).std()
        df_features[f'global_active_power_rolling_min_{window}h'] = df_features['Global_active_power'].rolling(window=window).min()
        df_features[f'global_active_power_rolling_max_{window}h'] = df_features['Global_active_power'].rolling(window=window).max()
    
    # Difference features (rate of change)
    df_features['global_active_power_diff_1h'] = df_features['Global_active_power'].diff(1)
    df_features['global_active_power_diff_24h'] = df_features['Global_active_power'].diff(24)
    df_features['global_active_power_diff_168h'] = df_features['Global_active_power'].diff(168)
    
    # Ratio features
    df_features['power_ratio_reactive_active'] = df_features['Global_reactive_power'] / (df_features['Global_active_power'] + 1e-8)
    df_features['power_ratio_sub1_active'] = df_features['Sub_metering_1'] / (df_features['Global_active_power'] + 1e-8)
    df_features['power_ratio_sub2_active'] = df_features['Sub_metering_2'] / (df_features['Global_active_power'] + 1e-8)
    df_features['power_ratio_sub3_active'] = df_features['Sub_metering_3'] / (df_features['Global_active_power'] + 1e-8)
    
    # Interaction features
    df_features['hour_day_interaction'] = df_features['hour'] * df_features['day_of_week']
    df_features['hour_month_interaction'] = df_features['hour'] * df_features['month']
    df_features['voltage_power_interaction'] = df_features['Voltage'] * df_features['Global_active_power']
    
    # Polynomial features for key variables
    df_features['global_active_power_squared'] = df_features['Global_active_power'] ** 2
    df_features['voltage_squared'] = df_features['Voltage'] ** 2
    
    df_features = df_features.dropna()
    
    print(f"Enhanced features created: {df_features.shape[1]} total columns")
    print(f"New features added: {df_features.shape[1] - df.shape[1]}")
    
    return df_features


def prepare_training_data(df: pd.DataFrame, target_column: str, 
                         lookback_window: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare time series sequences for sequence-based models."""
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns].values
    y = df[target_column].values
    
    X_sequences = []
    y_sequences = []
    
    for i in range(lookback_window, len(X)):
        X_sequences.append(X[i-lookback_window:i])
        y_sequences.append(y[i])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print(f"Training data prepared: {X_sequences.shape[0]} sequences")
    print(f"Feature shape: {X_sequences.shape}")
    print(f"Target shape: {y_sequences.shape}")
    
    return X_sequences, y_sequences


def prepare_training_data_flat(df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare flat data for tree-based models without sequences."""
    """
    Prepare training data for tree-based models (XGBoost, Decision Tree) without sequences.
    This is more appropriate for these models as they don't need temporal structure.
    """
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns].values
    y = df[target_column].values
    
    print(f"Flat training data prepared: {X.shape[0]} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def select_features(X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, k: int = 50) -> Tuple[np.ndarray, np.ndarray, SelectKBest]:
    """Select top k features using mutual information for regression."""
    """
    Select top k features using mutual information for regression.
    This helps reduce overfitting and improves model performance.
    """
    print(f"Selecting top {k} features from {X_train.shape[1]} total features...")
    
    # Use mutual information for feature selection (better for non-linear relationships)
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Feature selection completed: {X_train_selected.shape[1]} features selected")
    
    return X_train_selected, X_test_selected, selector


def split_data(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets chronologically."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]
    
    print(f"Data split:")
    print(f"Train: {train_data.shape[0]} rows ({train_data.index.min()} to {train_data.index.max()})")
    print(f"Validation: {val_data.shape[0]} rows ({val_data.index.min()} to {val_data.index.max()})")
    print(f"Test: {test_data.shape[0]} rows ({test_data.index.min()} to {test_data.index.max()})")
    
    return train_data, val_data, test_data