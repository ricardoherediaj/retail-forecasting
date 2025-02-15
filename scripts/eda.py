"""
Exploratory Data Analysis (EDA) module.
Handles statistical analysis and visualization of categorical and numerical features.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple

from paths import TRANSFORMED_DATA_DIR, ASSETS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """
    Load and combine categorical and numerical datasets from the transformed data directory.
    
    Returns:
        DataFrame: The combined dataset with categorical and numerical features
    """
    try:
        # Load categorical and numerical data
        cat = pd.read_parquet(TRANSFORMED_DATA_DIR / 'cat_result_quality.parquet')
        num = pd.read_parquet(TRANSFORMED_DATA_DIR / 'num_result_quality.parquet')
        
        # Combine and reorganize columns
        df = (pd.concat([cat, num], axis=1)
              .drop(columns=['id'])
              [['store_id', 'item_id', 'd', 'year', 'month', 'wday', 'weekday', 
                'event_name_1', 'event_type_1', 'wm_yr_wk', 'sales', 'sell_price']])
        
        logger.info("Data loaded and combined successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_plot(filename: str) -> None:
    """
    Helper function to save plots with consistent formatting.
    
    Args:
        filename: Name of the file to save
    """
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / filename)
    plt.close()

def plot_categorical_analysis(df: pd.DataFrame, categorical_cols: List[str]) -> None:
    """
    Create horizontal bar plots for categorical variables.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical columns to analyze
    """
    try:
        rows = (len(categorical_cols) + 1) // 2
        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(16, rows * 6))
        axes = axes.flat

        for idx, col in enumerate(categorical_cols):
            df[col].value_counts().plot.barh(ax=axes[idx])
            axes[idx].set_title(col, fontsize=12, fontweight="bold")
            axes[idx].tick_params(labelsize=12)

        save_plot('categorical_analysis.png')
        logger.info("Categorical analysis completed")
    except Exception as e:
        logger.error(f"Error in categorical analysis: {str(e)}")
        raise

def plot_time_series(df: pd.DataFrame, store_id: Optional[str] = None) -> None:
    """
    Plot time series of sales, optionally filtered by store.
    
    Args:
        df: Input DataFrame
        store_id: Optional store ID to filter data
    """
    try:
        data = df if store_id is None else df[df.store_id == store_id]
        title_prefix = '' if store_id is None else f'Store {store_id} - '
        
        # Overall sales trend
        plt.figure(figsize=(20, 10))
        data.groupby('date')['sales'].sum().plot()
        plt.title(f'{title_prefix}Sales Trend')
        save_plot(f'sales_trend{"_" + store_id if store_id else ""}.png')
        
        # Sales by product
        data.groupby(['date', 'item_id'])['sales'].sum().unstack().plot(
            subplots=True, layout=(5, 2), sharex=False, figsize=(20, 30)
        )
        plt.suptitle(f'{title_prefix}Sales by Product Tendency', y=1.02)
        save_plot(f'sales_by_product{"_" + store_id if store_id else ""}.png')
        
        logger.info(f"Time series plots generated for {store_id if store_id else 'all stores'}")
    except Exception as e:
        logger.error(f"Error in time series plotting: {str(e)}")
        raise

def plot_price_analysis(df: pd.DataFrame) -> None:
    """
    Plot price trends by product.
    """
    try:
        df.groupby(['date', 'item_id'])['sell_price'].mean().unstack().plot(
            subplots=True, layout=(5, 2), sharex=False, figsize=(20, 10)
        )
        plt.suptitle('Price Trend by Product', y=1.02)
        save_plot('price_trends.png')
        logger.info("Price analysis completed")
    except Exception as e:
        logger.error(f"Error in price analysis: {str(e)}")
        raise

def plot_seasonality(df: pd.DataFrame) -> None:
    """
    Plot various seasonality analyses by product.
    """
    try:
        # Monthly seasonality
        df.groupby(['month', 'item_id'])['sales'].mean().unstack().plot.bar(
            subplots=True, layout=(6, 2), sharex=False, figsize=(20, 30)
        )
        plt.suptitle('Monthly Seasonality by Product', y=1.02)
        save_plot('monthly_seasonality.png')
        
        # Weekly seasonality
        df.groupby(['weekday', 'item_id'])['sales'].mean().unstack().plot.bar(
            subplots=True, layout=(6, 2), sharex=False, figsize=(20, 30)
        )
        plt.suptitle('Weekly Seasonality by Product', y=1.02)
        save_plot('weekly_seasonality.png')
        
        logger.info("Seasonality analysis completed")
    except Exception as e:
        logger.error(f"Error in seasonality analysis: {str(e)}")
        raise

def plot_event_analysis(df: pd.DataFrame) -> None:
    """
    Plot event-related analyses by product.
    """
    try:
        # Event name analysis
        df.groupby(['event_name_1', 'item_id'])['sales'].mean().unstack().plot.barh(
            subplots=True, layout=(6, 2), sharex=False, figsize=(20, 40)
        )
        plt.suptitle('Event Name Seasonality by Product', y=1.02)
        save_plot('event_name_analysis.png')
        
        # Event type analysis
        df.groupby(['event_type_1', 'item_id'])['sales'].mean().unstack().plot.barh(
            subplots=True, layout=(6, 2), sharex=False, figsize=(20, 40)
        )
        plt.suptitle('Event Type Seasonality by Product', y=1.02)
        save_plot('event_type_analysis.png')
        
        logger.info("Event analysis completed")
    except Exception as e:
        logger.error(f"Error in event analysis: {str(e)}")
        raise

def main():
    """Execute the EDA pipeline."""
    try:
        # Create assets directory if it doesn't exist
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load data
        df = load_data()
        
        # Generate all plots
        categorical_cols = ['store_id', 'year', 'month', 'wday', 'weekday', 
                          'event_name_1', 'event_type_1']
        plot_categorical_analysis(df, categorical_cols)
        
        # Time series analysis
        plot_time_series(df)  # Overall
        plot_time_series(df, 'CA_3')  # Store CA_3
        plot_time_series(df, 'CA_4')  # Store CA_4
        
        # Price analysis
        plot_price_analysis(df)
        
        # Seasonality analysis
        plot_seasonality(df)
        
        # Event analysis
        plot_event_analysis(df)
        
        logger.info("EDA completed successfully")
        
    except Exception as e:
        logger.error(f"Error in EDA pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 