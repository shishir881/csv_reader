import pandas as pd
import numpy as np
import warnings

class DataInspector:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col

    def sanity_check(self):
        """Step 0: Fail fast if data is garbage."""
        if len(self.df) < 20: 
            return False, "Dataset too small (needs 20+ rows) for meaningful ML."
        
        if self.target_col not in self.df.columns:
            return False, f"Target column '{self.target_col}' not found in dataset."
            
        if self.df[self.target_col].nunique() <= 1:
            return False, "Target column has no variation (all values are the same)."
            
        return True, "Data looks good."

    def detect_problem_type(self):
        """
        Determines if this is a Regression or Classification problem.
        """
        target = self.df[self.target_col]
        
        # 1. If target is string/object/categorical -> CLASSIFICATION
        if target.dtype == 'object' or target.dtype.name == 'category':
            return 'classification'
            
        # 2. If target is numeric but has few unique values (e.g., 0 and 1) -> CLASSIFICATION
        # Rule: If unique values are < 10% of total rows AND count is low (<20)
        unique_count = target.nunique()
        if unique_count < 20 and (unique_count / len(self.df)) < 0.1:
            return 'classification'
            
        # 3. Default -> REGRESSION
        return 'regression'

    def get_column_types(self):
        """Step 1: Automatically categorize columns."""
        # Drop target from input features so we don't use it as input
        df = self.df.drop(columns=[self.target_col], errors='ignore')
        
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = []

        # Try to detect date columns automatically
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[col], errors='raise')
                    date_cols.append(col)
                    # If it's a date, remove from cat_cols
                    if col in cat_cols: cat_cols.remove(col)
                except (ValueError, TypeError):
                    pass
                    
        return {
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "date_cols": date_cols
        }