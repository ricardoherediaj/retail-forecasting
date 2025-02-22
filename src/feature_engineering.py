"""
Feature engineering module for retail forecasting project.
Handles creation of lag features, rolling windows, and categorical encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from typing import List, Optional, Dict
from pathlib import Path
import logging

from paths import TRANSFORMED_DATA_DIR

# Configuration parameters
DEFAULT_CONFIG = {
    'lag_periods': list(range(1, 16)),  # 15 days of lags
    'rolling_windows': list(range(2, 16)),  # 2-15 day windows
    'stock_break_periods': [3, 7, 15],
    'min_samples_leaf': 100,
    'columns_to_drop': ['d', 'wm_yr_wk', 'sell_price', 'stock_break_3',
                       'stock_break_7', 'stock_break_15'],
    'categorical_cols': ['year', 'month', 'wday', 'weekday',
                        'event_name_1', 'event_type_1']
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stock_break(sales: pd.Series, n: int = 5) -> np.ndarray:
    """
    Calculate stock break indicator for a series of sales.
    A stock break is defined as having n consecutive days with zero sales.

    Args:
        sales: Series of sales values
        n: Number of consecutive days to consider for stock break

    Returns:
        Array with 1s indicating stock breaks and 0s otherwise
    """
    zero_sales = pd.Series(np.where(sales == 0, 1, 0))
    num_zeros = zero_sales.rolling(n).sum()
    # Shift the result by 1 position forward to mark the last zero in the sequence
    return np.where((num_zeros == n) & (zero_sales == 1), 1, 0)

def create_stock_break_features(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """
    Create stock break indicators for different periods.

    Args:
        df: Input dataframe containing sales data
        periods: List of periods to check for stock breaks

    Returns:
        DataFrame with stock break indicators
    """
    if periods is None:
        periods = DEFAULT_CONFIG['stock_break_periods']

    result = pd.DataFrame()

    for period in periods:
        result[f'stock_break_{period}'] = (
            df.groupby(['store_id', 'item_id'])
            .sales.transform(lambda x: stock_break(x, period))
            .values
        )

    logger.info(f"Created stock break features for periods: {periods}")
    return result

def create_lag_features(df: pd.DataFrame,
                       columns: List[str],
                       lag_periods: List[int] = None) -> pd.DataFrame:
    """
    Create lag features for specified columns.
    Lag features help capture temporal dependencies in the data.

    Args:
        df: Input dataframe
        columns: List of columns to create lags for
        lag_periods: List of periods to lag by

    Returns:
        DataFrame with lag features
    """
    if lag_periods is None:
        lag_periods = DEFAULT_CONFIG['lag_periods']

    result = pd.DataFrame()

    for col in columns:
        for lag in lag_periods:
            result[f'{col}_lag_{lag}'] = df.groupby(['store_id', 'item_id'])[col].shift(lag)

    logger.info(f"Created lag features for columns: {columns}")
    return result

def create_rolling_features(df: pd.DataFrame,
                          column: str,
                          windows: List[int] = None,
                          stats: List[str] = ['min', 'mean', 'max']) -> pd.DataFrame:
    """
    Create rolling window features with specified statistics.
    Rolling features help smooth out short-term fluctuations and highlight trends.

    Args:
        df: Input dataframe
        column: Column to create rolling features for
        windows: List of window sizes
        stats: List of statistics to compute

    Returns:
        DataFrame with rolling window features
    """
    if windows is None:
        windows = DEFAULT_CONFIG['rolling_windows']

    result = pd.DataFrame()

    for window in windows:
        grouped = df.groupby(['store_id', 'item_id'])[column].shift(1).rolling(window)

        if 'min' in stats:
            result[f'{column}_minm_{window}'] = grouped.min()
        if 'mean' in stats:
            result[f'{column}_mm_{window}'] = grouped.mean()
        if 'max' in stats:
            result[f'{column}_maxm_{window}'] = grouped.max()

    logger.info(f"Created rolling features for column: {column}")
    return result

def encode_categorical_features(df: pd.DataFrame,
                              columns: List[str],
                              encoding: str = 'onehot',
                              target_col: Optional[str] = None) -> pd.DataFrame:
    """
    Encode categorical features using specified encoding.
    Supports both one-hot encoding and target encoding.

    Args:
        df: Input dataframe
        columns: Categorical columns to encode
        encoding: Type of encoding ('onehot' or 'target')
        target_col: Target column for target encoding

    Returns:
        DataFrame with encoded features
    """
    if encoding == 'onehot':
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[columns])
        return pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(),
            index=df.index
        )
    else:  # target encoding
        if target_col is None:
            raise ValueError("target_col must be specified for target encoding")
        encoder = TargetEncoder(
            min_samples_leaf=DEFAULT_CONFIG['min_samples_leaf'],
            return_df=False
        )
        encoded = encoder.fit_transform(df[columns], df[target_col])
        return pd.DataFrame(
            encoded,
            columns=[f"{col}_te" for col in columns],
            index=df.index
        )

def engineer_features(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    target_col: str = 'sales'
) -> pd.DataFrame:
    """
    Main function to orchestrate the feature engineering process.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical columns to encode
        target_col: Name of the target column (default: 'sales')
        
    Returns:
        DataFrame with engineered features
    """
    # Initialize with key columns
    result = df[['d', 'store_id', 'item_id', target_col]].copy()  # Keep key columns

    if categorical_cols is None:
        categorical_cols = DEFAULT_CONFIG['categorical_cols']

    logger.info("Starting feature engineering process")

    # Convert categorical columns to object type
    df = df.astype({
        'weekday': 'object',
        'wday': 'object',
        'month': 'object',
        'year': 'object'
    })

    # Sort data by store, item and date
    df = df.sort_values(['store_id', 'item_id', 'date'])

    # Create lag features
    lag_features = create_lag_features(
        df,
        columns=['sales', 'sell_price'],
        lag_periods=DEFAULT_CONFIG['lag_periods']
    )

    # Create rolling window features
    rolling_features = create_rolling_features(
        df,
        column='sales',
        windows=DEFAULT_CONFIG['rolling_windows']
    )

    # Create stock break features
    stock_features = create_stock_break_features(
        df,
        periods=DEFAULT_CONFIG['stock_break_periods']
    )

    # Create categorical encodings
    onehot_features = encode_categorical_features(
        df,
        columns=categorical_cols,
        encoding='onehot'
    )

    target_features = encode_categorical_features(
        df,
        columns=categorical_cols,
        encoding='target',
        target_col=target_col
    )

    # Combine all features
    result = pd.concat([
        result,
        onehot_features,
        target_features,
        lag_features,
        rolling_features,
        stock_features
    ], axis=1)

    # Drop unnecessary columns
    result = result.drop(columns=DEFAULT_CONFIG['columns_to_drop'], errors='ignore')

    # Drop rows with missing values from lag/rolling features
    result = result.dropna()

    logger.info("Feature engineering process completed")
    return result

def run_feature_engineering(cat_input_path: Path,
                          num_input_path: Path,
                          output_path: Path,
                          config: Dict = None) -> None:
    """
    Run the complete feature engineering pipeline.

    Args:
        cat_input_path: Path to categorical input parquet file
        num_input_path: Path to numerical input parquet file
        output_path: Path to save transformed features
        config: Configuration dictionary
    """
    logger.info("Reading input data")

    # Read input data
    cat = pd.read_parquet(cat_input_path)
    num = pd.read_parquet(num_input_path)

    # Join dataframes
    columns_to_drop = ['id', 'dept_id', 'cat_id', 'state_id']
    df = (pd.concat([cat, num], axis=1)
          .drop(columns=columns_to_drop, errors='ignore'))
    
    # Reset index to make date a column
    df = df.reset_index()

    # Get categorical columns from config or default
    categorical_cols = config['categorical_cols'] if config and 'categorical_cols' in config else DEFAULT_CONFIG['categorical_cols']

    # Run feature engineering
    df_transformed = engineer_features(
        df,
        categorical_cols=categorical_cols,
        target_col='sales'
    )

    # Save transformed features
    logger.info(f"Saving transformed features to {output_path}")
    df_transformed.to_parquet(output_path)
    logger.info("Feature engineering pipeline completed successfully")

def main():
    """Run the feature engineering pipeline."""
    try:
        logger.info("Starting feature engineering pipeline")
        
        # Define input and output paths
        cat_input = TRANSFORMED_DATA_DIR / 'cat_result_quality.parquet'
        num_input = TRANSFORMED_DATA_DIR / 'num_result_quality.parquet'
        output = TRANSFORMED_DATA_DIR / 'df_transformed.parquet'
        
        # Run feature engineering pipeline
        run_feature_engineering(
            cat_input_path=cat_input,
            num_input_path=num_input,
            output_path=output
        )
        
        logger.info("Feature engineering pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    main()