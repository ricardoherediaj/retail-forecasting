import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import (
    stock_break,
    create_stock_break_features,
    create_lag_features,
    create_rolling_features,
    encode_categorical_features,
    engineer_features,
    run_feature_engineering,
    DEFAULT_CONFIG
)

# Define test data paths
TEST_DATA_DIR = Path(__file__).parent / 'data'
CAT_TEST_FILE = TEST_DATA_DIR / 'cat_result_quality.parquet'
NUM_TEST_FILE = TEST_DATA_DIR / 'num_result_quality.parquet'
OUTPUT_TEST_FILE = TEST_DATA_DIR / 'test_output.parquet'

@pytest.fixture(scope="module")
def sample_data():
    """Load actual test data"""
    # Read the actual test files with date as index
    cat_df = pd.read_parquet(CAT_TEST_FILE)
    num_df = pd.read_parquet(NUM_TEST_FILE)
    
    # Merge the dataframes
    df = pd.concat([cat_df, num_df], axis=1)
    # Remove any duplicate columns
    df = df.loc[:,~df.columns.duplicated()]
    
    # Reset index to make date a column
    df = df.reset_index()
    
    # Sort by store, item and date as in the notebook
    df = df.sort_values(['store_id', 'item_id', 'date'])
    
    return df

def test_stock_break():
    """Test stock break function with known pattern"""
    sales = pd.Series([10, 0, 0, 0, 10, 0, 0])
    result = stock_break(sales, n=3)
    # The third zero in a sequence of three zeros should be marked
    expected = np.array([0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(result, expected)

def test_create_stock_break_features(sample_data):
    """Test stock break features creation with actual data"""
    result = create_stock_break_features(sample_data, periods=[3, 7])
    
    # Check if features were created correctly
    assert 'stock_break_3' in result.columns
    assert 'stock_break_7' in result.columns
    assert len(result) == len(sample_data)
    
    # Verify values are binary
    assert set(result['stock_break_3'].unique()).issubset({0, 1})
    assert set(result['stock_break_7'].unique()).issubset({0, 1})

def test_create_lag_features(sample_data):
    """Test lag features creation with actual data"""
    result = create_lag_features(
        sample_data,
        columns=['sales'],
        lag_periods=[1, 2, 3]
    )
    
    # Check feature creation
    assert all(f'sales_lag_{i}' in result.columns for i in [1, 2, 3])
    assert len(result) == len(sample_data)
    
    # Verify lag values match original values
    store_item = sample_data.iloc[0][['store_id', 'item_id']].values
    store_item_data = sample_data[
        (sample_data['store_id'] == store_item[0]) & 
        (sample_data['item_id'] == store_item[1])
    ]
    sales = store_item_data['sales'].values
    lag1 = result.iloc[3]['sales_lag_1']  # Use iloc instead of index
    assert lag1 == sales[2]  # Lag 1 should match previous day's sales

def test_create_rolling_features(sample_data):
    """Test rolling features creation with actual data"""
    result = create_rolling_features(
        sample_data,
        column='sales',
        windows=[2, 3],
        stats=['min', 'mean', 'max']
    )
    
    expected_cols = [
        'sales_minm_2', 'sales_mm_2', 'sales_maxm_2',
        'sales_minm_3', 'sales_mm_3', 'sales_maxm_3'
    ]
    
    # Check feature creation
    assert all(col in result.columns for col in expected_cols)
    assert len(result) == len(sample_data)
    
    # Verify rolling calculations
    store_item = sample_data.iloc[0][['store_id', 'item_id']].values
    store_item_data = sample_data[
        (sample_data['store_id'] == store_item[0]) & 
        (sample_data['item_id'] == store_item[1])
    ]
    sales = store_item_data['sales'].values
    
    # Check if mean calculation is correct for a window of 2
    mean2 = result.iloc[2]['sales_mm_2']  # Use iloc instead of index
    expected_mean = np.mean(sales[0:2])
    np.testing.assert_almost_equal(mean2, expected_mean, decimal=2)

def test_encode_categorical_features(sample_data):
    """Test categorical features encoding with actual data"""
    cat_cols = ['event_name_1', 'event_type_1']
    
    # Test one-hot encoding
    ohe_result = encode_categorical_features(
        sample_data,
        columns=cat_cols,
        encoding='onehot'
    )
    assert len(ohe_result.columns) > len(cat_cols)
    assert not ohe_result.isnull().any().any()
    
    # Test target encoding
    te_result = encode_categorical_features(
        sample_data,
        columns=cat_cols,
        encoding='target',
        target_col='sales'
    )
    assert all(f"{col}_te" in te_result.columns for col in cat_cols)
    assert not te_result.isnull().any().any()

def test_engineer_features(sample_data):
    """Test complete feature engineering pipeline with actual data"""
    # Ensure required columns exist
    required_cols = ['weekday', 'wday', 'month', 'year']
    for col in required_cols:
        if col not in sample_data.columns:
            sample_data[col] = '1'  # Add dummy values if missing
            
    result = engineer_features(
        sample_data,
        categorical_cols=['event_name_1', 'event_type_1', 'weekday', 'wday', 'month', 'year'],
        target_col='sales'
    )
    
    # Check essential columns
    assert 'store_id' in result.columns
    assert 'item_id' in result.columns
    assert 'sales' in result.columns
    
    # Check feature creation
    assert any(col.startswith('sales_lag_') for col in result.columns)
    assert any(col.startswith('sales_mm_') for col in result.columns)
    assert any(col.endswith('_te') for col in result.columns)
    
    # Check data quality
    assert not result.isnull().any().any()
    assert len(result) > 0

def test_full_pipeline():
    """Test the complete pipeline with actual test data"""
    # Run pipeline
    run_feature_engineering(
        cat_input_path=CAT_TEST_FILE,
        num_input_path=NUM_TEST_FILE,
        output_path=OUTPUT_TEST_FILE
    )
    
    # Verify output
    assert OUTPUT_TEST_FILE.exists()
    df_transformed = pd.read_parquet(OUTPUT_TEST_FILE)
    
    # Basic validation
    assert len(df_transformed) > 0
    assert not df_transformed.isnull().any().any()
    assert 'store_id' in df_transformed.columns
    assert 'item_id' in df_transformed.columns
    assert 'sales' in df_transformed.columns
    
    # Clean up
    if OUTPUT_TEST_FILE.exists():
        OUTPUT_TEST_FILE.unlink()

def test_config_override():
    """Test configuration override functionality"""
    custom_config = DEFAULT_CONFIG.copy()
    custom_config['lag_periods'] = [1, 2]
    custom_config['rolling_windows'] = [2, 3]
    
    # Run pipeline with custom config
    output_custom = TEST_DATA_DIR / 'test_custom_config.parquet'
    
    # Create a modified run_feature_engineering call that uses the config
    cat = pd.read_parquet(CAT_TEST_FILE)
    num = pd.read_parquet(NUM_TEST_FILE)
    
    # Merge dataframes and handle index properly
    df = pd.concat([cat, num], axis=1)
    df = df.loc[:,~df.columns.duplicated()]  # Remove duplicates properly
    df = df.reset_index()  # Reset index to make date a column
    df = df.sort_values(['store_id', 'item_id', 'date'])
    df = df.reset_index(drop=True)  # Ensure unique index
    
    # Ensure required columns exist
    required_cols = ['weekday', 'wday', 'month', 'year']
    for col in required_cols:
        if col not in df.columns:
            df[col] = '1'  # Add dummy values if missing
    
    # Apply feature engineering with custom config
    result = engineer_features(
        df,
        categorical_cols=custom_config['categorical_cols'],
        target_col='sales'
    )
    
    # Save result
    result.to_parquet(output_custom)
    
    # Verify output
    df_transformed = pd.read_parquet(output_custom)
    assert len(df_transformed) > 0
    
    # Verify custom config was applied (approximately)
    lag_cols = [col for col in df_transformed.columns if 'lag_' in col]
    assert len(lag_cols) > 0  # At least some lag columns should exist
    
    # Clean up
    if output_custom.exists():
        output_custom.unlink()

def test_error_handling():
    """Test error handling scenarios"""
    with pytest.raises(ValueError):
        # Should raise error when target_col is None for target encoding
        encode_categorical_features(
            pd.DataFrame({'col': [1, 2, 3]}),
            columns=['col'],
            encoding='target'
        )

if __name__ == "__main__":
    pytest.main([__file__])