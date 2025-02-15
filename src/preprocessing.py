"""
Preprocessing module for retail forecasting project.
Handles data cleaning, type conversion, and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from paths import TRANSFORMED_DATA_DIR

def load_data() -> pd.DataFrame:
    """Load the work dataset."""
    return pd.read_parquet(TRANSFORMED_DATA_DIR / 'work.parquet')

def convert_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert temporal columns to categorical."""
    categorical_cols = ['year', 'month', 'wday']
    df[categorical_cols] = df[categorical_cols].astype('category')
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    # Remove event_name_2 and event_type_2 due to high missingness
    df.drop(columns=['event_name_2', 'event_type_2'], inplace=True)
    
    # Fill missing events with 'No_event'
    event_cols = ['event_name_1', 'event_type_1']
    df[event_cols] = df[event_cols].fillna('No_event')
    
    # Handle missing sell_price values by imputing mode by product
    def impute_mode_by_product(group):
        mode_price = group['sell_price'].mode()[0]
        group.loc[group.sell_price.isna(), 'sell_price'] = mode_price
        return group
    
    # Select only the relevant columns after grouping
    df = df.groupby('item_id', group_keys=False).apply(impute_mode_by_product)
    return df

def split_numeric_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into numeric and categorical features."""
    cat = df.select_dtypes(exclude='number').copy()  # Categorical features
    num = df.select_dtypes(include='number').copy()  # Numeric features
    return num, cat

def save_processed_data(df: pd.DataFrame, num: pd.DataFrame, cat: pd.DataFrame):
    """Save processed datasets."""
    processed_work = TRANSFORMED_DATA_DIR / 'work_result_quality.parquet'
    processed_num = TRANSFORMED_DATA_DIR / 'num_result_quality.parquet'
    processed_cat = TRANSFORMED_DATA_DIR / 'cat_result_quality.parquet'
    
    df.to_parquet(processed_work)
    num.to_parquet(processed_num)
    cat.to_parquet(processed_cat)

def main():
    """Run the preprocessing pipeline."""
    try:
        # Load data
        df = load_data()
        
        # Convert categorical
        df = convert_categorical(df)
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Split numeric and categorical
        num, cat = split_numeric_categorical(df)
        
        # Save processed data
        save_processed_data(df, num, cat)
        
        print("Preprocessing completed successfully")
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 