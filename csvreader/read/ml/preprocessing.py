import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import entropy

class FeatureEngineer:
    def __init__(self, df, target_col=None, date_col=None, problem_type='regression'):
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.problem_type = problem_type
        
        self.dropped_leakage = []
        self.dropped_noise = []
        self.dropped_redundant = []

        self.reference_date = None  # Dataset ko sabai vanda paila ko date
        self.trend_freq = None      # Daily, Monthly, or Yearly

    @property
    def dropped_cols(self):
        return self.dropped_leakage + self.dropped_noise + self.dropped_redundant

    def _calculate_normalized_mi(self, df, target):
        """Calculates Mutual Information with Robust NaN Handling."""
        if df.empty: return pd.Series(dtype=float)
        X = df.drop(columns=[target])
        y = df[target]
        if X.shape[1] == 0: return pd.Series(dtype=float)

        # 1. FILL NaNs (Critical Fix)
        # For MI calculation only, we fill NaNs with -1 or 0 so sklearn doesn't crash
        X_filled = X.fillna(-1) 

        # 2. Encode inputs
        X_encoded = X_filled.copy()
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
        
        discrete_features = X_encoded.dtypes == int

        try:
            if self.problem_type == 'classification':
                if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
                    y_encoded = LabelEncoder().fit_transform(y)
                else:
                    y_encoded = y
                mi_scores = mutual_info_classif(X_encoded, y_encoded, discrete_features=discrete_features, random_state=42)
                
                target_entropy = entropy(pd.Series(y_encoded).value_counts(normalize=True))
                normalized_mi = np.zeros(len(mi_scores)) if target_entropy == 0 else mi_scores / target_entropy

            else: # Regression
                y_numeric = pd.to_numeric(y, errors='coerce').fillna(0)
                mi_scores = mutual_info_regression(X_encoded, y_numeric, discrete_features=discrete_features, random_state=42)
                normalized_mi = np.sqrt(1 - np.exp(-2 * mi_scores))
                
            normalized_mi = np.clip(normalized_mi, 0, 1)
            return pd.Series(normalized_mi, index=X.columns)

        except Exception as e:
            # RETRY BLOCK: Try without discrete_features + Ensure Float Types
            try:
                print(f"⚠️ Standard MI failed ({e}). Retrying with robust settings...")
                if self.problem_type == 'regression':
                     y_numeric = pd.to_numeric(y, errors='coerce').fillna(0)
                     # Force convert to float to avoid type issues
                     X_float = X_encoded.astype(float)
                     mi_scores = mutual_info_regression(X_float, y_numeric, random_state=42)
                     normalized_mi = np.sqrt(1 - np.exp(-2 * mi_scores))
                     normalized_mi = np.clip(normalized_mi, 0, 1)
                     return pd.Series(normalized_mi, index=X.columns)
            except:
                pass 

            # FINAL FALLBACK: Return 0.5 (Neutral) so features are KEPT
            print(f"⚠️ MI Calculation Completely Failed. Using safe fallback (0.5).")
            return pd.Series(0.5, index=X.columns)

    def filter_features(self, high_threshold=0.95, low_threshold=0.0):
        if self.df.empty: return self.df
        if not self.target_col or self.target_col not in self.df.columns: return self.df

        # 1. Calculate MI Scores
        mi_scores = self._calculate_normalized_mi(self.df, self.target_col)
        if mi_scores.empty: return self.df
        
        print("   -> MI Scores:")
        print(mi_scores.sort_values(ascending=False))

        # 2. Identify Bad Features
        leakage = mi_scores[mi_scores > high_threshold].index.tolist()
        if leakage:
            self.dropped_leakage.extend(leakage)

        noise = mi_scores[mi_scores < low_threshold].index.tolist()
        if noise:
            self.dropped_noise.extend(noise)

        # 3. Apply Drops
        cols_to_drop = list(set(self.dropped_leakage + self.dropped_noise))
        self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # 4. Redundancy Check
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            redundant = [column for column in upper_tri.columns if any(upper_tri[column] > high_threshold)]
            
            if redundant:
                self.dropped_redundant.extend(redundant)
                self.df.drop(columns=redundant, inplace=True, errors='ignore')

        return self.df

    def preprocess(self):
        # 1. Date Logic
        if self.date_col and self.date_col in self.df.columns:
            try:
                self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
                self.df = self.df.sort_values(by=self.date_col)
                
                # Starting date lai save garne (Future prediction ma kaam aauchha)
                self.reference_date = self.df[self.date_col].min()
                
                # Dui wota date bich ko average gap (din ma) nikalne
                avg_gap_days = self.df[self.date_col].diff().dt.days.median()
                
                # TIMLE VANEKO LOGIC: Check frequency and create 'time_index'
                if pd.isna(avg_gap_days):
                     self.df['time_index'] = 0
                elif avg_gap_days >= 25 and avg_gap_days <= 35:
                    # Monthly trend: Din random vaye pani ignore garne, mahina ko sequence banaune
                    self.trend_freq = 'monthly'
                    print("📅 Detected Monthly Trend. Sequencing by Month/Year...")
                    self.df['time_index'] = (self.df[self.date_col].dt.year - self.reference_date.year) * 12 + (self.df[self.date_col].dt.month - self.reference_date.month)
                elif avg_gap_days >= 360:
                    # Yearly trend
                    self.trend_freq = 'yearly'
                    print("📅 Detected Yearly Trend. Sequencing by Year...")
                    self.df['time_index'] = self.df[self.date_col].dt.year - self.reference_date.year
                else:
                    # Daily or Continuous trend
                    self.trend_freq = 'daily'
                    print("📅 Detected Daily/Continuous Trend. Sequencing by Days...")
                    self.df['time_index'] = (self.df[self.date_col] - self.reference_date).dt.days

                # Main model ko lagi basic year ra month pani rakhne
                self.df['year'] = self.df[self.date_col].dt.year
                self.df['month'] = self.df[self.date_col].dt.month
                
                # Sklearn crash nahos vanera original date column lai faldine
                self.df.drop(columns=[self.date_col], inplace=True)
                
            except Exception as e:
                print(f"⚠️ Date Processing Failed: {e}. Skipping advanced date features.")

        # 2. Target Cleaning
        if self.target_col and self.target_col in self.df.columns:
            self.df.dropna(subset=[self.target_col], inplace=True)
            if self.problem_type == 'regression':
                self.df[self.target_col] = pd.to_numeric(self.df[self.target_col], errors='coerce')
                self.df.dropna(subset=[self.target_col], inplace=True)
        
        # 3. Filtering
        if self.target_col and self.target_col in self.df.columns:
            self.filter_features(high_threshold=0.97, low_threshold=0.0)

        return self.df

    def get_sklearn_pipeline(self, num_cols, cat_cols):
        all_dropped = self.dropped_cols
        num_cols = [c for c in num_cols if c not in all_dropped]
        cat_cols = [c for c in cat_cols if c not in all_dropped]

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        return ColumnTransformer(transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])

    def get_future_time_index(user_date_str, reference_date, trend_freq):
        """
        User le haleko string date lai purano saved logic anusaar time_index ma convert garchha.
        """
        # 1. User ko date lai pandas DateTime ma convert garne
        future_date = pd.to_datetime(user_date_str)
        
        # 2. Trend Frequency anusaar Math lagayera time_index nikalne
        if trend_freq == 'monthly':
            # (Future Year - Purano Year) * 12 + (Future Month - Purano Month)
            time_index = (future_date.year - reference_date.year) * 12 + (future_date.month - reference_date.month)
        
        elif trend_freq == 'yearly':
            time_index = future_date.year - reference_date.year
            
        else: # 'daily' wa continuous
            time_index = (future_date - reference_date).days
            
        # User ko date bata year ra month pani firtà garne (Main prediction ma chaincha)
        return time_index, future_date.year, future_date.month