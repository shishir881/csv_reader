import pandas as pd
import joblib
import os

class BatchPredictor:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        """
        input_data: Can be a file path (str) OR a pandas DataFrame
        """
        try:
            # Handle CSV Path
            if isinstance(input_data, str):
                df = pd.read_csv(input_data)
            # Handle DataFrame directly
            elif isinstance(input_data, pd.DataFrame):
                df = input_data
            else:
                return None, "Invalid input format"

            output_df = df.copy()

            # Note: The 'pipeline' inside self.model handles the preprocessing (Encoding/Scaling)
            # So we pass the raw data directly to predict()
            
            try:
                predictions = self.model.predict(df)
            except Exception as e:
                return None, f"Prediction Logic Error: {str(e)}"

            output_df['predicted_result'] = predictions
            return output_df, "Success"
            
        except Exception as e:
            return None, f"General Error: {str(e)}"