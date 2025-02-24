"""
Test module for preprocessing.py
Tests data cleaning, type conversion, and feature engineering functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import src.preprocessing  # Add this import at the top level

from src.preprocessing import (
    load_data,
    convert_categorical,
    handle_missing_values,
    split_numeric_categorical,
    save_processed_data,
    main
)

@pytest.fixture
def sample_df():
    """Create a sample DataFrame that mirrors the structure of the actual data."""
    return pd.DataFrame({
        'year': [2021, 2021, 2022],
        'month': [1, 2, 3],
        'wday': [1, 2, 3],
        'item_id': ['ITEM1', 'ITEM1', 'ITEM2'],
        'event_name_1': ['Event1', None, 'Event2'],
        'event_type_1': ['Type1', None, 'Type2'],
        'event_name_2': ['Event3', None, None],
        'event_type_2': ['Type3', None, None],
        'sell_price': [10.0, None, 15.0],
        'sales': [100, 150, 200],
        'store_id': ['STORE1', 'STORE1', 'STORE2']
    })

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

def test_convert_categorical(sample_df):
    """Test conversion of temporal columns to categorical."""
    result = convert_categorical(sample_df.copy())
    
    categorical_cols = ['year', 'month', 'wday']
    for col in categorical_cols:
        assert result[col].dtype.name == 'category'
    
    # Verify other columns remain unchanged
    non_cat_cols = [col for col in result.columns if col not in categorical_cols]
    for col in non_cat_cols:
        assert result[col].dtype == sample_df[col].dtype

def test_handle_missing_values(sample_df):
    """Test handling of missing values in the dataset."""
    result = handle_missing_values(sample_df.copy())
    
    # Test event columns are filled with 'No_event'
    assert not result['event_name_1'].isna().any()
    assert not result['event_type_1'].isna().any()
    assert 'No_event' in result['event_name_1'].values
    assert 'No_event' in result['event_type_1'].values
    
    # Test event_name_2 and event_type_2 are dropped
    assert 'event_name_2' not in result.columns
    assert 'event_type_2' not in result.columns
    
    # Test sell_price is imputed by item_id
    assert not result['sell_price'].isna().any()
    # Check if mode imputation worked for ITEM1
    item1_price = result[result['item_id'] == 'ITEM1']['sell_price'].iloc[1]
    assert item1_price == 10.0  # Should be imputed with the mode

def test_split_numeric_categorical(sample_df):
    """Test splitting of data into numeric and categorical features."""
    num, cat = split_numeric_categorical(sample_df.copy())
    
    # Check numeric DataFrame
    assert all(num[col].dtype.kind in 'iuf' for col in num.columns)
    assert 'sales' in num.columns
    assert 'sell_price' in num.columns
    
    # Check categorical DataFrame
    assert all(cat[col].dtype.kind in 'OSU' for col in cat.columns)
    assert 'item_id' in cat.columns
    assert 'store_id' in cat.columns

def test_save_processed_data(sample_df, temp_data_dir, monkeypatch):
    """Test saving of processed datasets."""
    # Mock the TRANSFORMED_DATA_DIR
    monkeypatch.setattr('src.preprocessing.TRANSFORMED_DATA_DIR', temp_data_dir)  # Fixed the monkeypatch
    
    num, cat = split_numeric_categorical(sample_df.copy())
    save_processed_data(sample_df, num, cat)
    
    # Check if files were created
    assert (temp_data_dir / 'work_result_quality.parquet').exists()
    assert (temp_data_dir / 'num_result_quality.parquet').exists()
    assert (temp_data_dir / 'cat_result_quality.parquet').exists()
    
    # Verify data integrity
    saved_df = pd.read_parquet(temp_data_dir / 'work_result_quality.parquet')
    saved_num = pd.read_parquet(temp_data_dir / 'num_result_quality.parquet')
    saved_cat = pd.read_parquet(temp_data_dir / 'cat_result_quality.parquet')
    
    pd.testing.assert_frame_equal(saved_df, sample_df)
    pd.testing.assert_frame_equal(saved_num, num)
    pd.testing.assert_frame_equal(saved_cat, cat)

def test_main(sample_df, temp_data_dir, monkeypatch):
    """Test the main preprocessing pipeline."""
    # Mock dependencies
    def mock_load_data():
        return sample_df.copy()
    
    monkeypatch.setattr('src.preprocessing.load_data', mock_load_data)
    monkeypatch.setattr('src.preprocessing.TRANSFORMED_DATA_DIR', temp_data_dir)  # Fixed the monkeypatch
    
    # Run main
    main()
    
    # Verify all files were created
    assert (temp_data_dir / 'work_result_quality.parquet').exists()
    assert (temp_data_dir / 'num_result_quality.parquet').exists()
    assert (temp_data_dir / 'cat_result_quality.parquet').exists()

def test_main_error_handling(monkeypatch):
    """Test error handling in main function."""
    def mock_load_data_error():
        raise Exception("Test error")
    
    monkeypatch.setattr('src.preprocessing.load_data', mock_load_data_error)
    
    with pytest.raises(Exception) as exc_info:
        main()
    assert "Test error" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main([__file__]) 