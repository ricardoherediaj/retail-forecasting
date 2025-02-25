"""
Feature pipeline using Metaflow to orchestrate the feature engineering process.
This flow runs the entire pipeline from setup to variable preselection.
"""

from metaflow import FlowSpec, step, Parameter
import pandas as pd
from pathlib import Path
import sys
import os

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our existing modules
from setup import main as setup_main
from preprocessing import main as preprocess_main
from feature_engineering import main as feature_engineering_main
from variable_preselection import main as variable_preselection_main

class FeaturePipeline(FlowSpec):
    """
    A Metaflow pipeline that orchestrates the feature engineering process.
    This pipeline runs the following steps:
    1. Setup: Initial data loading and preparation
    2. Preprocessing: Data cleaning and type conversion
    3. Feature Engineering: Create features for modeling
    4. Variable Preselection: Select most important features
    """
    
    feature_selection_method = Parameter(
        'feature_selection_method',
        help='Method to use for feature selection (mi, rfe, or per)',
        default='mi'
    )

    @step
    def start(self):
        """Start the pipeline."""
        print("Starting feature engineering pipeline...")
        self.next(self.setup)

    @step
    def setup(self):
        """Run the setup process."""
        print("Running setup...")
        setup_main()
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """Run the preprocessing step."""
        print("Running preprocessing...")
        preprocess_main()
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """Run the feature engineering step."""
        print("Running feature engineering...")
        feature_engineering_main()
        self.next(self.variable_preselection)

    @step
    def variable_preselection(self):
        """Run the variable preselection step."""
        print("Running variable preselection...")
        variable_preselection_main(method=self.feature_selection_method)
        self.next(self.end)

    @step
    def end(self):
        """End the pipeline."""
        print("Feature pipeline completed successfully!")

if __name__ == '__main__':
    FeaturePipeline() 