import pandas as pd
import joblib
import os
from read.ml.trend_predictor import get_future_time_index
from read.ml.trend_predictor import AdvancedTrendPredictor

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
        
    
    # def predict_from_date(self, user_date_str, reference_date, trend_freq):
    #     """
    #     User le date matra halda yo function call hunchha.
    #     """
    #     if self.trend_predictor is None:
    #         return None, "Trend predictor is not available for this model. Please provide full feature data."
            
    #     try:
    #         # 1. Date lai time_index ma convert garne
    #         future_index, year, month = get_future_time_index(user_date_str, reference_date, trend_freq)
            
    #         # 2. Trend predictor use garera baki sabai features ko value nikalne
    #         predicted_features_df = self.trend_predictor.predict_for_date(future_index, year, month)
            
    #         # 3. Niskeko features lai main model ma pathayera final result nikalne
    #         final_prediction = self.model.predict(predicted_features_df)
            
    #         # Output dekhauna ko lagi sajilo format
    #         predicted_features_df['FINAL_PREDICTION'] = final_prediction
    #         return predicted_features_df, "Success"
            
    #     except Exception as e:
    #         return None, f"Error in date prediction: {str(e)}"