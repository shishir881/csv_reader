import pandas as pd
import numpy as np

class DataInspector:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col

    def sanity_check(self):
        if len(self.df) < 5: 
            return False, "Dataset too small (needs 5+ rows)."
        
        if self.target_col not in self.df.columns:
            return False, f"Target column '{self.target_col}' not found."
            
        if self.df[self.target_col].nunique() <= 1:
            return False, "Target column has no variation."
            
        return True, "Data looks good."

    def detect_problem_type(self):
        target = self.df[self.target_col]
        
        # 1. Convert to numeric
        target_numeric = pd.to_numeric(target, errors='coerce')
        valid_count = target_numeric.count()
        total_count = len(target)
        
        print(f"   [DEBUG] Valid Numeric Rows: {valid_count}/{total_count}")
        
        is_numeric_like = (valid_count / total_count) > 0.5
        
        if is_numeric_like:
            unique_count = target_numeric.nunique()
            ratio = unique_count / valid_count if valid_count > 0 else 0
            
            print(f"   [DEBUG] Unique Values: {unique_count}")
            print(f"   [DEBUG] Unique Ratio: {ratio:.4f}")

            # Check if values are Floats (Decimals)
            # If data is like 4.5, 7.1, it's almost certainly Regression
            is_float = False
            try:
                # Check if any value has a non-zero decimal part
                is_float = (target_numeric % 1 != 0).any()
            except:
                pass

            # RULE 1: It's Float data -> REGRESSION
            if is_float:
                print("   [DEBUG] Decision: REGRESSION (Detected Float/Decimals)")
                return 'regression'

            # RULE 2: High Cardinality (>20 unique values) -> REGRESSION
            # (Lowered from 50 to 20 to catch small datasets like yours)
            if unique_count > 20:
                print("   [DEBUG] Decision: REGRESSION (Unique > 20)")
                return 'regression'
                
            # RULE 3: Ratio Check
            if unique_count > 10 and ratio > 0.05:
                print("   [DEBUG] Decision: REGRESSION (Ratio > 5%)")
                return 'regression'
                
            print("   [DEBUG] Decision: CLASSIFICATION (Fallback)")
            return 'classification'
            
        return 'classification'

    def get_column_types(self):
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