"""
Test module for variable_preselection.py
Tests feature selection methods and data handling functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from src.variable_preselection import (
    load_data,
    split_xy,
    ranking_mi,
    select_mi_features,
    select_rfe_features,
    ranking_per,
    select_permutation_features,
    add_segmentation_variables,
    save_preselected_data,
    main,
    preprocess_features
)

# Configure test logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@pytest.fixture
def caplog_with_handler(caplog):
    """Fixture to ensure caplog captures all levels."""
    caplog.set_level(logging.INFO)
    return caplog

@pytest.fixture
def sample_df():
    """Create a sample DataFrame that mirrors the structure of the actual data."""
    # Create date range
    dates = pd.date_range(start='2013-01-01', periods=5)
    
    # Create sample data with the same structure as the real data
    df = pd.DataFrame({
        'date': dates,
        'store_id': ['STORE1'] * 5,
        'item_id': ['ITEM1'] * 5,
        'year_2013': [1.0] * 5,
        'year_2014': [0.0] * 5,
        'year_2015': [0.0] * 5,
        'month_1': [1.0] * 5,
        'month_2': [0.0] * 5,
        'sales': [100, 150, 200, 180, 220],
        'sales_lag_1': [np.nan, 100, 150, 200, 180],
        'sales_lag_2': [np.nan, np.nan, 100, 150, 200],
        'sales_mean_7': [100, 125, 150, 157.5, 170],
        'sales_std_7': [0, 35.36, 50, 41.93, 45.41],
        'event_name_1': ['Event1', None, 'Event2', None, 'Event3'],
        'event_type_1': ['Type1', None, 'Type2', None, 'Type3'],
        'event_name_2': ['Event4', None, None, None, None],
        'event_type_2': ['Type4', None, None, None, None],
        'sell_price': [10.0, None, 15.0, 12.0, 11.0]
    })
    
    # Handle missing values as in the main script
    df = preprocess_sample_df(df)
    return df

def preprocess_sample_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the sample DataFrame to handle missing values and encode categories."""
    # Remove event_name_2 and event_type_2 due to high missingness
    df = df.drop(columns=['event_name_2', 'event_type_2'])
    
    # Fill missing events with 'No_event'
    event_cols = ['event_name_1', 'event_type_1']
    df[event_cols] = df[event_cols].fillna('No_event')
    
    # Handle missing sell_price values by imputing mode by product
    df = impute_price_by_product(df)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in ['event_name_1', 'event_type_1']:
        df[col] = le.fit_transform(df[col])
    
    return df

def impute_price_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing prices using mode by product."""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Group by item_id and fill missing prices with mode
    price_modes = df.groupby('item_id')['sell_price'].transform(lambda x: x.mode().iloc[0])
    df.loc[df['sell_price'].isna(), 'sell_price'] = price_modes[df['sell_price'].isna()]
    
    return df

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

def preprocess_features(x: pd.DataFrame) -> pd.DataFrame:
    """Preprocess features for scikit-learn compatibility."""
    # Create a copy to avoid modifying the original
    x_processed = x.copy()
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    x_processed_array = imputer.fit_transform(x_processed)
    
    # Convert back to DataFrame
    x_processed = pd.DataFrame(x_processed_array, columns=x_processed.columns, index=x_processed.index)
    
    return x_processed

def test_split_xy(sample_df):
    """Test splitting data into features and target."""
    x, y = split_xy(sample_df)
    
    # Check that target column is not in features
    assert 'sales' not in x.columns
    assert 'date' not in x.columns
    assert 'store_id' not in x.columns
    assert 'item_id' not in x.columns
    
    # Check that we have the correct target
    assert isinstance(y, pd.Series)
    assert y.name == 'sales'
    assert len(y) == len(sample_df)

def test_ranking_mi(sample_df):
    """Test mutual information ranking functionality."""
    x, y = split_xy(sample_df)
    x_processed = preprocess_features(x)
    mutual_selector = mutual_info_regression(x_processed, y)
    
    rank = ranking_mi(x, mutual_selector)
    
    assert isinstance(rank, pd.DataFrame)
    assert all(col in rank.columns for col in ['variable', 'importance_mi', 'ranking_mi'])
    assert rank['importance_mi'].is_monotonic_decreasing
    assert len(rank) == len(x.columns)

def test_select_mi_features(sample_df):
    """Test mutual information feature selection."""
    x, y = split_xy(sample_df)
    x_processed = preprocess_features(x)
    x_selected = select_mi_features(x_processed, y, variable_limit_position=3)
    
    assert isinstance(x_selected, pd.DataFrame)
    assert len(x_selected.columns) <= 3
    assert len(x_selected) == len(x)

def test_select_rfe_features(sample_df):
    """Test recursive feature elimination."""
    x, y = split_xy(sample_df)
    x_processed = preprocess_features(x)
    x_selected = select_rfe_features(x_processed, y)
    
    assert isinstance(x_selected, pd.DataFrame)
    assert len(x_selected) == len(x)
    assert len(x_selected.columns) <= len(x.columns)

def test_select_permutation_features(sample_df):
    """Test permutation importance feature selection."""
    x, y = split_xy(sample_df)
    x_processed = preprocess_features(x)
    x_selected = select_permutation_features(x_processed, y, variable_limit_position=3)
    
    assert isinstance(x_selected, pd.DataFrame)
    assert len(x_selected.columns) <= 3
    assert len(x_selected) == len(x)

def test_add_segmentation_variables(sample_df):
    """Test adding segmentation variables back to the dataset."""
    x, y = split_xy(sample_df)
    x_processed = preprocess_features(x)
    x_selected = select_mi_features(x_processed, y, variable_limit_position=3)
    x_with_seg = add_segmentation_variables(x_selected, sample_df)
    
    assert 'date' in x_with_seg.columns
    assert 'store_id' in x_with_seg.columns
    assert 'item_id' in x_with_seg.columns
    assert len(x_with_seg) == len(sample_df)

def test_save_preselected_data(sample_df, temp_data_dir, monkeypatch):
    """Test saving preselected datasets."""
    monkeypatch.setattr('src.variable_preselection.TRANSFORMED_DATA_DIR', temp_data_dir)
    
    x, y = split_xy(sample_df)
    x_processed = preprocess_features(x)
    x_selected = select_mi_features(x_processed, y, variable_limit_position=3)
    x_with_seg = add_segmentation_variables(x_selected, sample_df)
    
    save_preselected_data(x_with_seg, y)
    
    # Check if files were created
    assert (temp_data_dir / 'x_preselection.parquet').exists()
    assert (temp_data_dir / 'y_preselection.parquet').exists()
    
    # Verify data integrity
    saved_x = pd.read_parquet(temp_data_dir / 'x_preselection.parquet')
    saved_y = pd.read_parquet(temp_data_dir / 'y_preselection.parquet')
    
    assert len(saved_x) == len(x_with_seg)
    assert len(saved_y) == len(y)
    assert 'sales' in saved_y.columns

def test_main_with_different_methods(sample_df, temp_data_dir, monkeypatch):
    """Test main function with different feature selection methods."""
    def mock_load_data():
        return sample_df.copy()
    
    monkeypatch.setattr('src.variable_preselection.load_data', mock_load_data)
    monkeypatch.setattr('src.variable_preselection.TRANSFORMED_DATA_DIR', temp_data_dir)
    
    # Test with each method
    for method in ['mi', 'rfe', 'per']:
        main(method=method)
        assert (temp_data_dir / 'x_preselection.parquet').exists()
        assert (temp_data_dir / 'y_preselection.parquet').exists()
        # Clean up for next iteration
        (temp_data_dir / 'x_preselection.parquet').unlink()
        (temp_data_dir / 'y_preselection.parquet').unlink()
    
    # Test with invalid method
    with pytest.raises(ValueError):
        main(method='invalid')

def test_main_error_handling(sample_df, temp_data_dir, monkeypatch):
    """Test error handling in main function."""
    def mock_load_data_error():
        raise Exception("Test error")
    
    monkeypatch.setattr('src.variable_preselection.load_data', mock_load_data_error)
    monkeypatch.setattr('src.variable_preselection.TRANSFORMED_DATA_DIR', temp_data_dir)
    
    with pytest.raises(Exception) as exc_info:
        main(method='mi')
    assert "Test error" in str(exc_info.value)

def test_logging_functionality(sample_df, temp_data_dir, monkeypatch, caplog):
    """Test logging functionality in main function."""
    def mock_load_data():
        return sample_df.copy()
    
    monkeypatch.setattr('src.variable_preselection.load_data', mock_load_data)
    monkeypatch.setattr('src.variable_preselection.TRANSFORMED_DATA_DIR', temp_data_dir)
    
    # Set logging level to INFO
    caplog.set_level(logging.INFO)
    
    # Test successful execution
    main(method='mi')
    
    # Check for expected log messages
    assert any("Starting variable preselection" in record.message for record in caplog.records)
    assert any("Loading data" in record.message for record in caplog.records)
    assert any("Splitting data into features and target" in record.message for record in caplog.records)
    
    # Clear logs for next test
    caplog.clear()
    
    # Test error case
    def mock_load_data_error():
        raise Exception("Test error")
    
    monkeypatch.setattr('src.variable_preselection.load_data', mock_load_data_error)
    
    with pytest.raises(Exception):
        main(method='mi')
    
    # Check for error log
    assert any("Error in variable preselection" in record.message for record in caplog.records)

if __name__ == "__main__":
    pytest.main([__file__]) 