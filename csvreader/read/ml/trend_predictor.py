import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier

def get_future_time_index(user_date_str, reference_date, trend_freq):
    """
    User le haleko string date lai purano saved logic anusaar time_index, year, ra month ma convert garchha.
    """
    try:
        future_date = pd.to_datetime(user_date_str)
        
        if trend_freq == 'monthly':
            time_index = (future_date.year - reference_date.year) * 12 + (future_date.month - reference_date.month)
        elif trend_freq == 'yearly':
            time_index = future_date.year - reference_date.year
        else: # 'daily' wa continuous
            time_index = (future_date - reference_date).days
            
        return time_index, future_date.year, future_date.month
    except Exception as e:
        print(f"Date conversion error: {e}")
        return 0, 2024, 1 # Fallback safe values


class AdvancedTrendPredictor:
    def __init__(self):
        self.models = {}             
        self.scaler = StandardScaler() 
        self.expected_features = []  
        self.time_cols = []          

    def train_trends(self, df, expected_features):
        """
        Main model le use garne features haru ko list liyera harek ko lagi mini-model train garne.
        df: preprocessing.py bata aayeko clean DataFrame
        """
        self.expected_features = expected_features
        
        # preprocessing.py le nikalne time columns
        possible_time_cols = ['time_index', 'year', 'month'] 
        
        # Dataset ma je je time columns available chhan, tyo matra line
        self.time_cols = [c for c in possible_time_cols if c in df.columns]
        
        if not self.time_cols:
            print("⚠️ No time columns found! Trend prediction might fail.")
            return

        # 1. Feature Engineering: X (Input) lai Scale garne
        X_raw = df[self.time_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_raw)

        # 2. Harek feature ko lagi chhutai model banaune
        for col in expected_features:
            if col in self.time_cols or col == 'day':
                continue 
            
            if col not in df.columns:
                continue

            y = df[col]
            
            # --- NUMERIC FEATURE LOGIC (Ridge Regression for Trend Extrapolation) ---
            if pd.api.types.is_numeric_dtype(y):
                y_clean = y.fillna(y.median()) 
                
                # Ridge Regression le continuous future trend smoothly predict garchha
                model = Ridge(alpha=1.0)
                model.fit(X_scaled, y_clean)
                
                self.models[col] = {'type': 'numeric', 'model': model}

            # --- CATEGORICAL FEATURE LOGIC (KNN Classification) ---
            else:
                y_clean = y.fillna(y.mode()[0] if not y.mode().empty else 'Unknown')
                
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_clean.astype(str))
                
                # KNN Classifier le pattern herera category choose garchha
                model = KNeighborsClassifier(n_neighbors=min(5, len(y_clean)))
                model.fit(X_scaled, y_encoded)
                
                self.models[col] = {'type': 'categorical', 'model': model, 'le': le}

    def predict_for_date(self, future_time_index, year, month):
        """
        future_time_index, year, month pass garda expected features ko row return garchha (DataFrame ko rup ma).
        """
        input_data = {}
        if 'time_index' in self.time_cols: input_data['time_index'] = [future_time_index]
        if 'year' in self.time_cols: input_data['year'] = [year]
        if 'month' in self.time_cols: input_data['month'] = [month]

        X_future = pd.DataFrame(input_data)
        
        # Scaling the future input perfectly like training data
        X_future_scaled = self.scaler.transform(X_future)
        
        predicted_row = {}
        
        for col in self.expected_features:
            if col in self.time_cols:
                predicted_row[col] = X_future[col].iloc[0]
                continue
                
            if col in self.models:
                info = self.models[col]
                
                if info['type'] == 'numeric':
                    # Ridge Prediction
                    pred = info['model'].predict(X_future_scaled)[0]
                    predicted_row[col] = round(pred, 4)
                else:
                    # KNN Categorical Prediction
                    pred_encoded = info['model'].predict(X_future_scaled)[0]
                    pred_text = info['le'].inverse_transform([pred_encoded])[0]
                    predicted_row[col] = pred_text
            else:
                predicted_row[col] = np.nan 
                
        return pd.DataFrame([predicted_row])