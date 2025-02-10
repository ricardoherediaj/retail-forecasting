# preprocessing.py

from pathlib import Path
import os
import numpy as np
import pandas as pd
import sqlalchemy as sa

# Import paths from paths.py
from paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR

def create_output_directory(output_dir=TRANSFORMED_DATA_DIR):
    """
    Create the output directory if it does not exist.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_data(db_path=str(RAW_DATA_DIR / 'supermarket.db')):
    """
    Connect to the SQLite database and load the data from three tables:
    'calendar', 'sales', and 'sell_prices'.
    """
    engine = sa.create_engine('sqlite:///' + db_path)
    # Load data from the three tables
    calendar = pd.read_sql("SELECT * FROM calendar", engine)
    sales = pd.read_sql("SELECT * FROM sales", engine)
    prices = pd.read_sql("SELECT * FROM sell_prices", engine)

    # Drop the 'index' column if it exists
    for df in [calendar, sales, prices]:
        if 'index' in df.columns:
            df.drop(columns='index', inplace=True)
    return calendar, sales, prices

def transform_sales(sales):
    """
    Transform the 'sales' DataFrame from wide to long format using melt.
    """
    sales_long = sales.melt(
        id_vars=sales.columns[:6],
        var_name='d',
        value_name='ventas'
    )
    if 'id' in sales_long.columns:
        sales_long.drop(columns='id', inplace=True)
    return sales_long

def merge_data(sales, calendar, prices):
    """
    Merge the sales, calendar, and prices DataFrames into an analytical table.
    First, merge sales with calendar on 'd' and then merge with prices on
    ['store_id', 'item_id', 'wm_yr_wk'].
    """
    df = sales.merge(calendar, how='left', on='d')
    df = df.merge(prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])
    return df

def reorder_and_index(df):
    """
    Reorder the columns of the DataFrame and set 'date' as the index.

    The column order is defined based on the teacher's notebook.
    """
    columns_order = [
        'date', 'state_id', 'store_id', 'dept_id', 'cat_id', 'item_id',
        'wm_yr_wk', 'd', 'ventas', 'sell_price', 'year', 'month', 'wday',
        'weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
    ]
    df = df[columns_order]
    df.set_index('date', inplace=True)
    return df

def split_and_save_datasets(df, output_dir=TRANSFORMED_DATA_DIR):
    """
    Split the DataFrame into validation and training sets based on date ranges,
    then save them as CSV files.

    Validation: Records from 2015-12-01 to 2015-12-31.
    Training: Records up to 2015-11-30.
    """
    validation = df.loc['2015-12-01':'2015-12-31']
    training = df.loc[:'2015-11-30']

    validation_path = output_dir / 'validation.csv'
    training_path = output_dir / 'training.csv'

    validation.to_csv(str(validation_path))
    training.to_csv(str(training_path))

    return validation_path, training_path

def impute_mode(group):
    """
    Impute missing 'sell_price' values in a group by the mode (most frequent value).
    """
    mode_value = group['sell_price'].mode()[0]
    group.loc[group['sell_price'].isna(), 'sell_price'] = mode_value
    return group

def data_quality_checks(df):
    """
    Perform data quality checks and cleaning on the DataFrame.

    Steps include:
    - Converting 'year', 'month', and 'wday' to categorical data (object type).
    - Dropping columns that are not relevant.
    - Handling missing values in categorical columns.
    - Imputing missing 'sell_price' values by grouping by 'item_id'.

    Returns:
        Cleaned DataFrame, categorical DataFrame, and numerical DataFrame.
    """
    # Convert columns to categorical type
    df = df.astype({'year': 'object', 'month': 'object', 'wday': 'object'})

    # Drop irrelevant columns
    df.drop(columns=['state_id', 'cat_id', 'dept_id'], inplace=True)

    # Separate the DataFrame into categorical and numerical parts
    cat = df.select_dtypes(exclude='number').copy()
    num = df.select_dtypes(include='number').copy()

    # Handle missing values in categorical columns
    for col in ['event_name_2', 'event_type_2']:
        if col in cat.columns:
            cat.drop(columns=col, inplace=True)

    # Impute missing categorical values with 'Sin_evento'
    for col in ['event_name_1', 'event_type_1']:
        if col in cat.columns:
            cat[col] = cat[col].fillna('Sin_evento')

    # Impute missing numerical values in 'sell_price' grouped by 'item_id'
    if 'item_id' in cat.columns:
        num = pd.concat([num, cat['item_id']], axis=1)
        num = num.groupby('item_id', group_keys=False).apply(impute_mode)
        num.drop(columns='item_id', inplace=True)

    return df, cat, num

def save_quality_datasets(df, cat, num, output_dir=TRANSFORMED_DATA_DIR):
    """
    Save the cleaned DataFrame, categorical DataFrame, and numerical DataFrame as pickle files.
    """
    df_path = output_dir / 'clean_df.pickle'
    cat_path = output_dir / 'clean_cat.pickle'
    num_path = output_dir / 'clean_num.pickle'

    df.to_pickle(str(df_path))
    cat.to_pickle(str(cat_path))
    num.to_pickle(str(num_path))

    return df_path, cat_path, num_path

# Uncomment the following block to run the full preprocessing pipeline directly.
# When using this module in your Notebook, import the functions
# and execute them cell by cell.
#
# if __name__ == '__main__':
#     # Create output directory if it doesn't exist
#     output_dir = create_output_directory()
#
#     # Load data from the database (data/raw/supermarket.db)
#     calendar, sales, prices = load_data()
#
#     # Transform the sales data from wide to long format
#     sales_long = transform_sales(sales)
#
#     # Merge the datasets to create the analytical table
#     df_merged = merge_data(sales_long, calendar, prices)
#
#     # Reorder the columns and set 'date' as index
#     df_ordered = reorder_and_index(df_merged)
#
#     # Split into validation and training sets and save as CSV in data/transformed
#     val_path, train_path = split_and_save_datasets(df_ordered, output_dir)
#
#     # (Optional) Perform data quality checks on the training set
#     training_df = pd.read_csv(str(train_path), parse_dates=['date'], index_col='date')
#     clean_df, clean_cat, clean_num = data_quality_checks(training_df)
#     quality_paths = save_quality_datasets(clean_df, clean_cat, clean_num, output_dir)
#
#     print("Data preprocessing and quality checks completed.")