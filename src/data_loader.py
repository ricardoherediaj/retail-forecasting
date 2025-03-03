"""
Data loader module for retail forecasting project.

This module handles the initial data loading and basic validation steps for the retail
forecasting pipeline. It provides functions to:
1. Connect to the SQLite database
2. Load and transform the calendar, sales, and pricing data
3. Perform basic data validation
4. Create training and validation datasets
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict
import sys

# Add parent directory to path to import paths module
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import inspect

# Import project paths
from paths import (
    RAW_DATA_DIR,
)

def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir (Path): Directory where log files will be stored
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_dir / 'data_loader.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Get the src directory and set up logging
src_dir = Path(__file__).parent
logger = setup_logging(src_dir / 'logs')

def get_project_paths() -> Dict[str, Path]:
    """
    Get the project's important paths.
    
    Returns:
        Dict[str, Path]: Dictionary containing project paths
    """
    # Get the src directory (where this file is located)
    src_dir = Path(__file__).parent
    temp_dir = src_dir / 'temp'
    
    paths = {
        'raw_data': RAW_DATA_DIR,
        'temp': temp_dir,  # All outputs will go to temp directory
    }
    
    # Create temp directory if it doesn't exist
    paths['temp'].mkdir(parents=True, exist_ok=True)
        
    logger.debug(f"Project paths initialized: {paths}")
    return paths

def connect_to_database(database_path: Path) -> sa.engine.Engine:
    """
    Create a connection to the SQLite database.
    
    Args:
        database_path (Path): Path to the SQLite database file
    
    Returns:
        sa.engine.Engine: SQLAlchemy engine instance
    
    Raises:
        FileNotFoundError: If the database file doesn't exist
    """
    logger.info(f"Attempting to connect to database at: {database_path}")
    
    if not database_path.exists():
        logger.error(f"Database file not found at: {database_path}")
        raise FileNotFoundError(f"Database file not found at {database_path}")
    
    engine = sa.create_engine(f'sqlite:///{str(database_path)}')
    logger.info("Database connection established successfully")
    return engine

def load_raw_tables(engine: sa.engine.Engine) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw tables from the database.
    
    Args:
        engine (sa.engine.Engine): SQLAlchemy engine instance
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Calendar, sales, and pricing data
    """
    logger.info("Starting to load tables from database")
    
    logger.debug("Loading calendar table")
    calendar_df = pd.read_sql('calendar', engine)
    
    logger.debug("Loading sales table")
    sales_df = pd.read_sql('sales', engine)
    
    logger.debug("Loading prices table")
    prices_df = pd.read_sql('sell_prices', engine)
    
    # Drop any index columns that might have been added during database creation
    for df in [calendar_df, sales_df, prices_df]:
        if 'index' in df.columns:
            logger.debug(f"Dropping 'index' column from DataFrame")
            df.drop(columns='index', inplace=True)
    
    logger.info("All tables loaded successfully")
    logger.debug(f"Table shapes - Calendar: {calendar_df.shape}, Sales: {sales_df.shape}, Prices: {prices_df.shape}")
    
    # Log preview of each table
    logger.info("\nCalendar data preview:")
    logger.info("\n" + str(calendar_df.head()))
    logger.info("\nSales data preview:")
    logger.info("\n" + str(sales_df.head()))
    logger.info("\nPrices data preview:")
    logger.info("\n" + str(prices_df.head()))
    
    return calendar_df, sales_df, prices_df

def transform_sales_data(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform sales data from wide to long format.
    
    Args:
        sales_df (pd.DataFrame): Sales data in wide format
    
    Returns:
        pd.DataFrame: Transformed sales data in long format
    """
    logger.info("Transforming sales data from wide to long format")
    
    # Identify ID columns (all columns except d_* columns)
    id_columns = sales_df.columns[:6].tolist()
    
    # Melt the dataframe to convert d_* columns to rows
    sales_melted = sales_df.melt(
        id_vars=id_columns,
        var_name='d',
        value_name='sales'
    )
    
    # Drop the 'id' column as per setup notebook
    sales_melted.drop(columns='id', inplace=True)
    
    logger.info("Sales data transformation completed")
    logger.debug(f"Transformed sales shape: {sales_melted.shape}")
    logger.info("\nTransformed sales data preview:")
    logger.info("\n" + str(sales_melted.head()))
    
    return sales_melted

def merge_tables(sales_df: pd.DataFrame, calendar_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all tables following the relationships defined in the data model.
    
    Args:
        sales_df (pd.DataFrame): Sales data
        calendar_df (pd.DataFrame): Calendar data
        prices_df (pd.DataFrame): Pricing data
    
    Returns:
        pd.DataFrame: Merged dataset
    """
    logger.info("Starting table merging process")
    
    # First merge: sales + calendar
    logger.debug("Merging sales with calendar data")
    df = sales_df.merge(right=calendar_df, how='left', on='d')
    
    # Second merge: add prices
    logger.debug("Merging with pricing data")
    df = df.merge(
        right=prices_df,
        how='left',
        on=['store_id', 'item_id', 'wm_yr_wk']
    )
    
    # Reorder columns as per setup notebook
    columns_order = [
        'date', 'state_id', 'store_id', 'dept_id', 'cat_id', 'item_id',
        'wm_yr_wk', 'd', 'sales', 'sell_price', 'year', 'month', 'wday',
        'weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
    ]
    
    df = df[columns_order].set_index('date')
    
    logger.info("Table merging completed successfully")
    logger.debug(f"Final merged shape: {df.shape}")
    logger.info("\nMerged data preview:")
    logger.info("\n" + str(df.head()))
    
    return df

def create_train_val_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and validation sets.
    
    Training data: All data up to November 30th, 2015
    Validation data: All data from December 1st, 2015 onwards
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        tuple: (training_df, validation_df)
    """
    logger.info("Creating training/validation split")
    
    training = df[:'2015-11-30']
    validation = df['2015-12-01':]
    
    logger.info(f"Split completed - Training shape: {training.shape}, Validation shape: {validation.shape}")
    logger.info("\nTraining data preview:")
    logger.info("\n" + str(training.head()))
    logger.info("\nValidation data preview:")
    logger.info("\n" + str(validation.head()))
    
    return training, validation

def save_datasets(training_df: pd.DataFrame, validation_df: pd.DataFrame, paths: Dict[str, Path]) -> None:
    """
    Save training and validation datasets to temp directory.
    
    Args:
        training_df (pd.DataFrame): Training dataset
        validation_df (pd.DataFrame): Validation dataset
        paths (Dict[str, Path]): Dictionary of project paths
    """
    logger.info("Saving datasets to temp directory")
    
    # Save validation dataset
    val_path = paths['temp'] / 'validation.parquet'
    validation_df.to_parquet(val_path)
    logger.info(f"Validation dataset saved to: {val_path}")
    
    # Save training dataset
    train_path = paths['temp'] / 'work.parquet'
    training_df.to_parquet(train_path)
    logger.info(f"Training dataset saved to: {train_path}")

def process_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to process all data.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation datasets
    """
    logger.info("Starting data processing pipeline")
    
    # Get paths and connect to database
    paths = get_project_paths()
    db_path = paths['raw_data'] / 'supermarket.db'
    engine = connect_to_database(db_path)
    
    # Load raw data
    calendar_df, sales_df, prices_df = load_raw_tables(engine)
    
    # Transform sales data
    sales_transformed = transform_sales_data(sales_df)
    
    # Merge all tables
    merged_df = merge_tables(sales_transformed, calendar_df, prices_df)
    
    # Create train/validation split
    training_df, validation_df = create_train_val_split(merged_df)
    
    # Save datasets
    save_datasets(training_df, validation_df, paths)
    
    logger.info("Data processing pipeline completed successfully")
    return training_df, validation_df

if __name__ == "__main__":
    # Set up file logging when run as main script
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('data_loader.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    try:
        logger.info("Starting main script execution")
        training_df, validation_df = process_data()
        logger.info("Script execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main script execution: {str(e)}", exc_info=True)
        raise 