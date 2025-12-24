import pandas as pd
import numpy as np

class DataInspector:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col

    def sanity_check(self):
        """Step 0: Fail fast if data is garbage."""
        # 1. Check size
        if len(self.df) < 50:
            return False, "Dataset too small (needs 50+ rows)."
        
        # 2. Check target variation
        if self.df[self.target_col].nunique() <= 1:
            return False, "Target column has no variation (all values are the same)."
            
        return True, "Data looks good."

    def get_column_types(self):
        """Step 1: Automatically categorize columns."""
        df = self.df.drop(columns=[self.target_col], errors='ignore')
        
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = []

        # Try to detect date columns automatically
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
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