"""
Tests for the data loader module.
"""

import pytest
import pandas as pd
import sqlalchemy as sa
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path to import data_loader
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import (
    get_project_paths,
    connect_to_database,
    load_raw_tables,
    transform_sales_data,
    merge_tables,
    create_train_val_split,
    save_datasets,
    process_data
)


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create minimal sample data for testing.
    
    Returns:
        tuple: A tuple containing (calendar_df, sales_df, prices_df)
    """
    # Simple sales data with just two days
    sales_df = pd.DataFrame({
        'id': ['SALES_1', 'SALES_2'],
        'item_id': ['ITEM_1', 'ITEM_2'],
        'store_id': ['STORE_1', 'STORE_1'],
        'dept_id': ['DEPT_1', 'DEPT_1'],
        'cat_id': ['CAT_1', 'CAT_1'],
        'state_id': ['CA', 'CA'],
        'd_1': [10, 5],
        'd_2': [15, 8]
    })
    
    # Calendar data matching the sales days
    calendar_df = pd.DataFrame({
        'date': ['2015-11-30', '2015-12-01'],
        'd': ['d_1', 'd_2'],
        'wm_yr_wk': [11548, 11549],
        'year': [2015, 2015],
        'month': [11, 12],
        'wday': [1, 2],
        'weekday': ['Monday', 'Tuesday'],
        'event_name_1': [None, 'Christmas'],
        'event_type_1': [None, 'Holiday'],
        'event_name_2': [None, None],
        'event_type_2': [None, None]
    })
    
    # Prices data matching the items
    prices_df = pd.DataFrame({
        'store_id': ['STORE_1', 'STORE_1'],
        'item_id': ['ITEM_1', 'ITEM_2'],
        'wm_yr_wk': [11548, 11548],
        'sell_price': [9.99, 5.99]
    })
    
    return calendar_df, sales_df, prices_df


def test_get_project_paths() -> None:
    """Test that project paths are correctly set up."""
    with patch('data_loader.RAW_DATA_DIR', Path('/fake/raw/data')):
        paths = get_project_paths()
        
        assert 'raw_data' in paths
        assert 'temp' in paths
        assert isinstance(paths['raw_data'], Path)
        assert isinstance(paths['temp'], Path)


def test_connect_to_database(tmp_path: Path) -> None:
    """Test database connection."""
    # Create a temporary SQLite database
    db_path = tmp_path / "test.db"
    engine = sa.create_engine(f'sqlite:///{db_path}')
    engine.connect()  # Create the file
    
    # Test the function
    result_engine = connect_to_database(db_path)
    assert isinstance(result_engine, sa.engine.Engine)


def test_connect_to_database_file_not_found() -> None:
    """Test that FileNotFoundError is raised when database doesn't exist."""
    with pytest.raises(FileNotFoundError):
        connect_to_database(Path('nonexistent.db'))


def test_load_raw_tables(tmp_path: Path) -> None:
    """Test loading data from SQLite database."""
    # Create a temporary SQLite database
    db_path = tmp_path / "test.db"
    engine = sa.create_engine(f'sqlite:///{db_path}')
    
    # Create test tables
    calendar_df = pd.DataFrame({'date': ['2015-11-30'], 'd': ['d_1']})
    sales_df = pd.DataFrame({'item_id': ['ITEM_1'], 'd_1': [10]})
    prices_df = pd.DataFrame({'store_id': ['STORE_1'], 'item_id': ['ITEM_1']})
    
    calendar_df.to_sql('calendar', engine, index=False)
    sales_df.to_sql('sales', engine, index=False)
    prices_df.to_sql('sell_prices', engine, index=False)
    
    # Test the function
    cal, sales, prices = load_raw_tables(engine)
    
    assert isinstance(cal, pd.DataFrame)
    assert isinstance(sales, pd.DataFrame)
    assert isinstance(prices, pd.DataFrame)
    assert len(cal) == 1
    assert len(sales) == 1
    assert len(prices) == 1


def test_transform_sales_data(sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """Test sales data transformation."""
    _, sales_df, _ = sample_data
    
    result = transform_sales_data(sales_df)
    
    assert isinstance(result, pd.DataFrame)
    assert 'sales' in result.columns
    assert 'd' in result.columns
    assert len(result) == 4  # 2 items × 2 days = 4 rows
    assert 'id' not in result.columns


def test_merge_tables(sample_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    """Test that data is correctly merged."""
    calendar_df, sales_df, prices_df = sample_data
    
    # First transform sales data
    sales_transformed = transform_sales_data(sales_df)
    
    # Then merge
    result = merge_tables(sales_transformed, calendar_df, prices_df)
    
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'date'
    assert 'sales' in result.columns
    assert 'sell_price' in result.columns
    assert len(result) == 4  # 2 items × 2 days = 4 rows


def test_create_train_val_split() -> None:
    """Test that data is correctly split into training and validation sets."""
    # Create test dataframe with dates as index
    dates = pd.date_range(start='2015-11-01', end='2015-12-31')
    df = pd.DataFrame({'sales': range(len(dates))}, index=dates)
    
    train_df, val_df = create_train_val_split(df)
    
    # Check that split is correct
    assert len(train_df) + len(val_df) == len(df)
    assert train_df.index.max().strftime('%Y-%m-%d') <= '2015-11-30'
    assert val_df.index.min().strftime('%Y-%m-%d') >= '2015-12-01'
    assert val_df.index.max().strftime('%Y-%m-%d') <= '2015-12-31'


def test_save_datasets(tmp_path: Path) -> None:
    """Test that data is correctly saved to parquet files."""
    # Create test dataframes
    train_df = pd.DataFrame({'sales': [1, 2, 3]})
    val_df = pd.DataFrame({'sales': [4, 5]})
    
    # Create paths dict
    paths = {'temp': tmp_path}
    
    # Save data
    save_datasets(train_df, val_df, paths)
    
    # Check that files were created
    assert (tmp_path / 'work.parquet').exists()
    assert (tmp_path / 'validation.parquet').exists()
    
    # Check that data can be loaded back
    train_loaded = pd.read_parquet(tmp_path / 'work.parquet')
    val_loaded = pd.read_parquet(tmp_path / 'validation.parquet')
    
    assert len(train_loaded) == len(train_df)
    assert len(val_loaded) == len(val_df)


@patch('data_loader.get_project_paths')
@patch('data_loader.connect_to_database')
@patch('data_loader.load_raw_tables')
@patch('data_loader.transform_sales_data')
@patch('data_loader.merge_tables')
@patch('data_loader.create_train_val_split')
@patch('data_loader.save_datasets')
def test_process_data(
    mock_save: MagicMock,
    mock_split: MagicMock,
    mock_merge: MagicMock,
    mock_transform: MagicMock,
    mock_load: MagicMock,
    mock_connect: MagicMock,
    mock_paths: MagicMock
) -> None:
    """Test the full data processing pipeline."""
    # Set up mocks
    mock_paths.return_value = {'raw_data': Path('/fake/raw'), 'temp': Path('/fake/temp')}
    mock_connect.return_value = 'engine'
    mock_load.return_value = ('calendar', 'sales', 'prices')
    mock_transform.return_value = 'transformed_sales'
    mock_merge.return_value = 'merged_df'
    mock_split.return_value = ('train_df', 'val_df')
    
    # Call the function
    train, val = process_data()
    
    # Check that all functions were called with correct arguments
    mock_paths.assert_called_once()
    mock_connect.assert_called_once_with(Path('/fake/raw/supermarket.db'))
    mock_load.assert_called_once_with('engine')
    mock_transform.assert_called_once_with('sales')
    mock_merge.assert_called_once_with('transformed_sales', 'calendar', 'prices')
    mock_split.assert_called_once_with('merged_df')
    mock_save.assert_called_once_with('train_df', 'val_df', {'raw_data': Path('/fake/raw'), 'temp': Path('/fake/temp')})
    
    # Check return values
    assert train == 'train_df'
    assert val == 'val_df'


def test_process_data_exception() -> None:
    """Test that exceptions in the pipeline are properly handled."""
    with patch('data_loader.get_project_paths', side_effect=Exception('Test error')):
        with pytest.raises(Exception) as excinfo:
            process_data()
        assert 'Test error' in str(excinfo.value) 