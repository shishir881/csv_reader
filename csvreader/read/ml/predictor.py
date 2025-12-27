import pandas as pd
import joblib
import os
from .preprocessing import FeatureEngineer 

class BatchPredictor:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, input_csv_path, date_col=None):
        try:
            df = pd.read_csv(input_csv_path)
        except Exception as e:
            return None, f"Failed to read CSV: {str(e)}"

        output_df = df.copy()

        # Apply SAME Engineering (Target=None means no filtering/lagging)
        engineer = FeatureEngineer(df, target_col=None, date_col=date_col) 
        X_new = engineer.preprocess() 
        
        try:
            predictions = self.model.predict(X_new)
        except Exception as e:
            return None, f"Prediction Error: {str(e)}"

        output_df['predicted_result'] = predictions
        return output_df, "Success"