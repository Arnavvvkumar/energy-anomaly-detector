import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


def plot_time_series(df: pd.DataFrame, columns: List[str], 
                    title: str = "Time Series Plot") -> None:
    """Plot time series data for specified columns."""
    plt.figure(figsize=(15, 8))
    
    for col in columns:
        if col in df.columns:
            plt.plot(df.index, df[col], label=col, alpha=0.7)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_distributions(df: pd.DataFrame, columns: List[str]) -> None:
    """Plot distribution histograms for specified columns."""
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(columns):
        if col in df.columns:
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {col}', fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_correlations(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """Plot correlation heatmap for numerical columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    correlation_matrix = df[columns].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_seasonal_patterns(df: pd.DataFrame, target_column: str) -> None:
    """Plot seasonal patterns and trends in the target variable."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    if 'hour' in df.columns:
        hourly_avg = df.groupby('hour')[target_column].mean()
        axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o')
        axes[0, 0].set_title('Average Power by Hour of Day', fontweight='bold')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Average Power (kW)')
        axes[0, 0].grid(True, alpha=0.3)
    
    if 'day_of_week' in df.columns:
        daily_avg = df.groupby('day_of_week')[target_column].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(range(7), daily_avg.values)
        axes[0, 1].set_title('Average Power by Day of Week', fontweight='bold')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Power (kW)')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(day_names)
        axes[0, 1].grid(True, alpha=0.3)
    
    if 'month' in df.columns:
        monthly_avg = df.groupby('month')[target_column].mean()
        axes[1, 0].plot(monthly_avg.index, monthly_avg.values, marker='o')
        axes[1, 0].set_title('Average Power by Month', fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Power (kW)')
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'quarter' in df.columns:
        quarterly_avg = df.groupby('quarter')[target_column].mean()
        axes[1, 1].bar(quarterly_avg.index, quarterly_avg.values)
        axes[1, 1].set_title('Average Power by Quarter', fontweight='bold')
        axes[1, 1].set_xlabel('Quarter')
        axes[1, 1].set_ylabel('Average Power (kW)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_trend_analysis(df: pd.DataFrame, target_column: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    axes[0].plot(df.index, df[target_column], alpha=0.7, linewidth=0.8)
    axes[0].set_title(f'{target_column} Over Time', fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Power (kW)')
    axes[0].grid(True, alpha=0.3)
    
    if len(df) > 1000:
        sample_size = min(1000, len(df))
        sample_indices = np.linspace(0, len(df)-1, sample_size, dtype=int)
        sample_df = df.iloc[sample_indices]
        
        axes[1].scatter(sample_df.index, sample_df[target_column], alpha=0.5, s=1)
        axes[1].set_title(f'{target_column} Scatter Plot (Sampled)', fontweight='bold')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Power (kW)')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    summary_stats = df[numeric_columns].describe()
    
    summary_stats.loc['missing_count'] = df[numeric_columns].isnull().sum()
    summary_stats.loc['missing_percent'] = (df[numeric_columns].isnull().sum() / len(df)) * 100
    
    print("Summary Statistics:")
    print("=" * 50)
    print(summary_stats.round(3))
    
    return summary_stats


def plot_feature_importance(df: pd.DataFrame, target_column: str) -> None:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    
    correlations = []
    for col in numeric_columns:
        if col in df.columns:
            corr = df[col].corr(df[target_column])
            correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    features, corr_values = zip(*correlations[:15])
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(features)), corr_values)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Absolute Correlation with Target')
    plt.title('Feature Importance (Correlation with Target)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{corr_values[i]:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()


def plot_missing_values(df: pd.DataFrame) -> None:
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percent': missing_percent
    }).sort_values('Missing Count', ascending=False)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(missing_df)), missing_df['Missing Percent'])
    plt.xticks(range(len(missing_df)), missing_df.index, rotation=45)
    plt.ylabel('Missing Values (%)')
    plt.title('Missing Values by Column', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars):
        if missing_df.iloc[i]['Missing Percent'] > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{missing_df.iloc[i]["Missing Percent"]:.1f}%', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_outlier_analysis(df: pd.DataFrame, columns: List[str]) -> None:
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(columns):
        if col in df.columns:
            plt.subplot(n_rows, n_cols, i + 1)
            plt.boxplot(df[col].dropna())
            plt.title(f'Box Plot of {col}', fontweight='bold')
            plt.ylabel(col)
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()