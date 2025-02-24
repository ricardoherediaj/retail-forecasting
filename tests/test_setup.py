import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sqlalchemy as sa
import sys
import os
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.setup import (
    connect_database,
    load_tables,
    merge_tables,
    create_time_splits,
    save_splits
)

from paths import RAW_DATA_DIR

@pytest.fixture(scope="module")
def test_engine():
    """Create a test database connection using the actual database."""
    db_path = RAW_DATA_DIR / 'supermarket.db'
    return sa.create_engine('sqlite:///' + str(db_path))

@pytest.fixture(scope="module")
def sample_tables(test_engine):
    """Load sample tables from the actual database."""
    return load_tables(test_engine)

def test_connect_database():
    """Test database connection using the actual database."""
    engine = connect_database(db_filename='supermarket.db')  # Use default path from setup.py
    assert isinstance(engine, sa.engine.Engine)
    
    # Test if we can query the database
    with engine.connect() as conn:
        result = conn.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result]
        assert all(table in tables for table in ['calendar', 'sales', 'sell_prices'])

def test_load_tables(test_engine):
    """Test loading tables from database."""
    calendar, sales, prices = load_tables(test_engine)
    
    # Check if DataFrames are returned
    assert all(isinstance(df, pd.DataFrame) for df in [calendar, sales, prices])
    
    # Check essential columns exist
    assert all(col in calendar.columns for col in ['date', 'd', 'wm_yr_wk'])
    assert all(col in sales.columns for col in ['item_id', 'store_id'])
    assert all(col in prices.columns for col in ['store_id', 'item_id', 'sell_price'])
    
    # Check no 'index' column exists
    assert all('index' not in df.columns for df in [calendar, sales, prices])

def test_merge_tables(sample_tables):
    """Test merging of tables."""
    calendar, sales, prices = sample_tables
    merged_df = merge_tables(calendar, sales, prices)
    
    # Check if result is a DataFrame
    assert isinstance(merged_df, pd.DataFrame)
    
    # Check if date is the index and is datetime
    assert merged_df.index.name == 'date'
    assert isinstance(merged_df.index, pd.DatetimeIndex)
    
    # Check if essential columns are present
    essential_cols = [
        'item_id', 'store_id', 'sales', 'wm_yr_wk',
        'weekday', 'month', 'year', 'sell_price'
    ]
    assert all(col in merged_df.columns for col in essential_cols)
    
    # Check if index is sorted
    assert merged_df.index.is_monotonic_increasing

def test_create_time_splits(sample_tables):
    """Test time-based splitting of data."""
    calendar, sales, prices = sample_tables
    merged_df = merge_tables(calendar, sales, prices)
    work, validation = create_time_splits(merged_df)
    
    # Check if both are DataFrames
    assert isinstance(work, pd.DataFrame)
    assert isinstance(validation, pd.DataFrame)
    
    # Check date ranges
    assert work.index.max() <= pd.Timestamp('2015-11-30')
    assert validation.index.min() >= pd.Timestamp('2015-12-01')
    assert validation.index.max() <= pd.Timestamp('2015-12-31')
    
    # Check no data overlap
    assert len(set(work.index) & set(validation.index)) == 0
    
    # Check all data is accounted for
    assert len(merged_df) == len(work) + len(validation)

def test_save_splits(tmp_path, sample_tables):
    """Test saving splits to parquet files."""
    # Get real data using the actual pipeline logic
    calendar, sales, prices = sample_tables
    merged_df = merge_tables(calendar, sales, prices)
    work, validation = create_time_splits(merged_df)
    
    # Create temporary directories
    temp_transformed = tmp_path / 'transformed'
    temp_validation = tmp_path / 'validation'
    temp_transformed.mkdir()
    temp_validation.mkdir()
    
    # Mock the directory constants
    import src.setup
    original_transformed = src.setup.TRANSFORMED_DATA_DIR
    original_validation = src.setup.VALIDATION_DIR
    src.setup.TRANSFORMED_DATA_DIR = temp_transformed
    src.setup.VALIDATION_DIR = temp_validation
    
    try:
        # Save splits
        save_splits(work, validation)
        
        # Check if files exist
        assert (temp_transformed / 'work.parquet').exists()
        assert (temp_validation / 'validation.parquet').exists()
        
        # Check if files can be read and contain correct data
        work_loaded = pd.read_parquet(temp_transformed / 'work.parquet')
        validation_loaded = pd.read_parquet(temp_validation / 'validation.parquet')
        
        # Verify data integrity
        assert len(work_loaded) == len(work)
        assert len(validation_loaded) == len(validation)
        assert work_loaded.index.max() <= pd.Timestamp('2015-11-30')
        assert validation_loaded.index.min() >= pd.Timestamp('2015-12-01')
        
    finally:
        # Restore original paths
        src.setup.TRANSFORMED_DATA_DIR = original_transformed
        src.setup.VALIDATION_DIR = original_validation

def test_full_pipeline():
    """Test the complete setup pipeline."""
    try:
        import src.setup
        src.setup.main()
        
        # Check if output files exist
        assert (src.setup.TRANSFORMED_DATA_DIR / 'work.parquet').exists()
        assert (src.setup.VALIDATION_DIR / 'validation.parquet').exists()
        
        # Load and verify the files
        work = pd.read_parquet(src.setup.TRANSFORMED_DATA_DIR / 'work.parquet')
        validation = pd.read_parquet(src.setup.VALIDATION_DIR / 'validation.parquet')
        
        # Basic validation
        assert len(work) > 0
        assert len(validation) > 0
        assert work.index.max() <= pd.Timestamp('2015-11-30')
        assert validation.index.min() >= pd.Timestamp('2015-12-01')
        
    except Exception as e:
        pytest.fail(f"Pipeline failed with error: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__]) 