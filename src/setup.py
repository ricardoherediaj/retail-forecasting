"""
Initial data setup module for retail forecasting project.
Handles database extraction, merging, and train/validation splitting.
"""

import pandas as pd
import sqlalchemy as sa
from pathlib import Path
import logging
from typing import Tuple

from paths import (
    RAW_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    VALIDATION_DIR
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def connect_database(db_filename: str = 'supermarket.db') -> sa.engine.Engine:
    """Create database connection."""
    database_path = RAW_DATA_DIR / db_filename
    return sa.create_engine('sqlite:///' + str(database_path))

def load_tables(engine: sa.engine.Engine) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw tables from database and clean column names.
    """
    logger.info("Loading tables from database...")
    
    # Load tables
    calendar = pd.read_sql('calendar', engine)
    sales = pd.read_sql('sales', engine)
    prices = pd.read_sql('sell_prices', engine)
    
    # Drop index columns if they exist
    for df in [calendar, sales, prices]:
        if 'index' in df.columns:
            df.drop(columns='index', inplace=True)
    
    return calendar, sales, prices

def merge_tables(calendar: pd.DataFrame, 
                sales: pd.DataFrame, 
                prices: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all tables into a single dataframe.
    
    Process:
    1. Unpivot sales data from wide to long format
    2. Merge with calendar data
    3. Merge with pricing data
    """
    logger.info("Merging tables...")
    
    # Unpivot sales data
    id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_long = pd.melt(
        sales,
        id_vars=id_columns,
        var_name='d',
        value_name='sales'
    )
    
    # Merge with calendar
    df = sales_long.merge(calendar, on='d', how='left')
    
    # Merge with prices
    df = df.merge(
        prices,
        on=['store_id', 'item_id', 'wm_yr_wk'],
        how='left'
    )
    
    # Convert date column to datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Sort index
    df.sort_index(inplace=True)
    
    logger.info(f"Final merged dataset shape: {df.shape}")
    return df

def create_time_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into work and validation sets.
    Validation set: December 2015
    Work set: All data before December 2015
    """
    logger.info("Creating time-based splits...")
    
    validation = df['2015-12-01':'2015-12-31']
    work = df[:'2015-11-30']  # Fixed date format
    
    logger.info(f"Work set shape: {work.shape}")
    logger.info(f"Validation set shape: {validation.shape}")
    
    return work, validation

def save_splits(work: pd.DataFrame, validation: pd.DataFrame):
    """
    Save work and validation datasets in parquet format.
    """
    logger.info("Saving datasets in parquet format...")
    
    # Create directories if they don't exist
    TRANSFORMED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    work.to_parquet(TRANSFORMED_DATA_DIR / 'work.parquet')
    validation.to_parquet(VALIDATION_DIR / 'validation.parquet')
    
    logger.info("Datasets saved successfully")

def main():
    """Run the initial data setup pipeline."""
    try:
        # Connect to database
        engine = connect_database()
        
        # Load raw tables
        calendar, sales, prices = load_tables(engine)
        
        # Merge tables
        df = merge_tables(calendar, sales, prices)
        
        # Create splits
        work, validation = create_time_splits(df)
        
        # Save splits
        save_splits(work, validation)
        
        logger.info("Initial data setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data setup: {str(e)}")
        raise

if __name__ == "__main__":
    main() 