"""
Variable preselection module for retail forecasting project.
Implements feature selection using supervised methods (Mutual Information, RFE, Permutation Importance).
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import warnings
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer
import multiprocessing

# Configure multiprocessing to avoid fork() warnings on macOS
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

from paths import TRANSFORMED_DATA_DIR

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers if they don't exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)

def load_data() -> pd.DataFrame:
    """Load the transformed dataset."""
    return pd.read_parquet(TRANSFORMED_DATA_DIR / 'df_transformed.parquet')

def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split data into features and target."""
    target = 'sales'
    
    # Columns to drop if they exist
    drop_cols = ['store_id', 'item_id', 'sales']
    existing_cols = [col for col in drop_cols if col in df.columns]
    
    x = df.drop(columns=existing_cols).copy()
    y = df[target].copy()
    return x, y

def ranking_mi(x: pd.DataFrame, mutual_selector: np.ndarray) -> pd.DataFrame:
    """Create ranking based on mutual information scores."""
    ranking_mi = pd.DataFrame(mutual_selector, index=x.columns).reset_index()
    ranking_mi.columns = ['variable', 'importance_mi']
    ranking_mi = ranking_mi.sort_values(by='importance_mi', ascending=False)
    ranking_mi['ranking_mi'] = np.arange(0, ranking_mi.shape[0])
    return ranking_mi

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

def select_mi_features(x: pd.DataFrame, y: pd.Series, variable_limit_position: int = 10) -> pd.DataFrame:
    """Select features using mutual information."""
    x_processed = preprocess_features(x)
    mutual_selector = mutual_info_regression(x_processed, y)
    rank = ranking_mi(x, mutual_selector)
    selected_features = rank['variable'].head(variable_limit_position).tolist()
    return x[selected_features]

def get_xgb_model() -> XGBRegressor:
    """Create and return an XGBoost model with standard configuration."""
    return XGBRegressor(
        n_jobs=1,  # Reduced parallelism to avoid warnings
        eval_metric='mae',
        random_state=42
    )

def select_rfe_features(x: pd.DataFrame, y: pd.Series, variable_limit_position: int = 10) -> pd.DataFrame:
    """Select features using recursive feature elimination."""
    x_processed = preprocess_features(x)
    model = get_xgb_model()
    rfe = RFE(estimator=model, n_features_to_select=variable_limit_position)
    rfe.fit(x_processed, y)
    selected_features = x.columns[rfe.support_].tolist()
    return x[selected_features]

def ranking_per(x: pd.DataFrame, permutation) -> pd.DataFrame:
    """Create ranking based on permutation importance scores."""
    ranking_per = pd.DataFrame({
        'variable': x.columns,
        'importance_per': permutation
    }).sort_values(by='importance_per', ascending=False)
    ranking_per['ranking_per'] = np.arange(0, ranking_per.shape[0])
    return ranking_per

def select_permutation_features(x: pd.DataFrame, y: pd.Series, variable_limit_position: int = 10) -> pd.DataFrame:
    """Select features using permutation importance."""
    x_processed = preprocess_features(x)
    model = get_xgb_model()
    model.fit(x_processed, y)
    
    # Use single thread for permutation importance to avoid warnings
    result = permutation_importance(
        model, x_processed, y,
        scoring='neg_mean_absolute_percentage_error',
        n_repeats=3,
        n_jobs=1,  # Single thread to avoid warnings
        random_state=42
    )
    
    rank = ranking_per(x, result.importances_mean)
    selected_features = rank['variable'].head(variable_limit_position).tolist()
    return x[selected_features]

def add_segmentation_variables(x_preselection: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add segmentation variables back to the dataset."""
    # Define segmentation columns
    seg_cols = ['store_id', 'item_id']
    
    # Only include columns that exist in the original DataFrame
    existing_seg_cols = [col for col in seg_cols if col in df.columns]
    
    return pd.concat([df[existing_seg_cols], x_preselection], axis=1)

def save_preselected_data(x_preselection: pd.DataFrame, y: pd.Series):
    """Save preselected datasets."""
    path_x_preselection = TRANSFORMED_DATA_DIR / 'x_preselection.parquet'
    path_y_preselection = TRANSFORMED_DATA_DIR / 'y_preselection.parquet'
    
    logger.info(f"Saving X preselection data to {path_x_preselection}")
    x_preselection.to_parquet(path_x_preselection)
    
    logger.info(f"Saving y preselection data to {path_y_preselection}")
    y_preselection_df = y.reset_index()
    y_preselection_df.columns = ['index', 'sales']
    y_preselection_df.to_parquet(path_y_preselection, index=False)

def main(method: str = 'mi'):
    """Run the variable preselection pipeline.
    
    Args:
        method: Feature selection method to use ('mi', 'rfe', or 'per')
    """
    try:
        logger.info(f"Starting variable preselection using {method} method")
        
        # Load data
        logger.info("Loading data...")
        df = load_data()
        
        # Split X and y
        logger.info("Splitting data into features and target...")
        x, y = split_xy(df)
        
        # Select features based on method
        logger.info(f"Selecting features using {method} method...")
        if method == 'mi':
            x_preselection = select_mi_features(x, y)
        elif method == 'rfe':
            x_preselection = select_rfe_features(x, y)
        elif method == 'per':
            x_preselection = select_permutation_features(x, y)
        else:
            logger.error(f"Unknown method: {method}")
            raise ValueError(f"Unknown method: {method}")
        
        # Add segmentation variables
        logger.info("Adding segmentation variables...")
        x_preselection = add_segmentation_variables(x_preselection, df)
        
        # Save results
        logger.info("Saving preselected data...")
        save_preselected_data(x_preselection, y)
        
        logger.info(f"Variable preselection completed successfully using {method} method")
        
    except Exception as e:
        logger.error(f"Error in variable preselection: {str(e)}")
        raise

if __name__ == "__main__":
    main() 