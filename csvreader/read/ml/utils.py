import pandas as pd
import numpy as np

class DataInspector:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col

    def sanity_check(self):
        """Step 0: Fail fast if data is garbage."""
        if len(self.df) < 5: 
            return False, "Dataset too small (needs 5+ rows) for meaningful ML."
        
        if self.target_col not in self.df.columns:
            return False, f"Target column '{self.target_col}' not found in dataset."
            
        if self.df[self.target_col].nunique() <= 1:
            return False, "Target column has no variation (all values are the same)."
            
        return True, "Data looks good."

    def detect_problem_type(self):
        """
        Determines if this is a Regression or Classification problem.
        Handles dirty data (e.g. numeric columns with '?' or strings).
        """
        target = self.df[self.target_col]
        
        # 1. Try to convert to numeric to see if it's actually a number
        # 'coerce' turns '?' into NaN so we can count valid numbers
        target_numeric = pd.to_numeric(target, errors='coerce')
        valid_count = target_numeric.count()
        total_count = len(target)
        
        # If > 50% of the data is numeric, treat it as numeric-like
        is_numeric_like = (valid_count / total_count) > 0.5
        
        if is_numeric_like:
            unique_count = target_numeric.nunique()
            # If unique values are many (>20 and >5% of valid rows) -> REGRESSION
            # (e.g. Prices, Temperatures, Sales)
            if unique_count > 20 and (unique_count / valid_count) > 0.05:
                return 'regression'
            # Otherwise -> CLASSIFICATION 
            # (e.g. 0, 1, 2 or Ratings 1-5)
            return 'classification'
            
        # 2. If it's strictly text/categorical -> CLASSIFICATION
        return 'classification'

    def get_column_types(self):
        """Step 1: Automatically categorize columns."""
        df = self.df.drop(columns=[self.target_col], errors='ignore')
        
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = []

        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_cols.append(col)
                    if col in cat_cols: cat_cols.remove(col)
                except (ValueError, TypeError):
                    pass
                    
        return {
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "date_cols": date_cols
        }