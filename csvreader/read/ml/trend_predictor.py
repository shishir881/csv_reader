import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier


def get_future_time_index(user_date_str, reference_date, trend_freq):
    """Convert a date string to (time_index, year, month) using the saved freq logic."""
    try:
        future_date = pd.to_datetime(user_date_str)
        if trend_freq == 'monthly':
            time_index = (
                (future_date.year - reference_date.year) * 12 +
                (future_date.month - reference_date.month)
            )
        elif trend_freq == 'yearly':
            time_index = future_date.year - reference_date.year
        else:  # daily / default
            time_index = (future_date - reference_date).days

        return time_index, future_date.year, future_date.month

    except Exception as e:
        print(f"   ⚠️  Date conversion error: {e}")
        return 0, 2024, 1


class AdvancedTrendPredictor:
    """
    Learns to predict each feature value from time_index / year / month.

    Input df (from FeatureEngineer.get_trend_training_df()) contains:
      - time_index, year, month   (time axes)
      - all training feature cols (what we want to predict)

    Safely skips columns that are all-NaN or have too few valid rows.
    """

    def __init__(self):
        self.models           = {}
        self.scaler           = StandardScaler()
        self.expected_features = []
        self.time_cols        = []
        self.is_trained       = False

    def train_trends(self, df, expected_features):
        """
        df              — output of FeatureEngineer.get_trend_training_df()
        expected_features — list of feature columns the main model was trained on
        """
        self.expected_features = expected_features

        possible_time_cols = ['time_index', 'year', 'month']
        self.time_cols = [c for c in possible_time_cols if c in df.columns]

        if not self.time_cols:
            print("   ⚠️  No time columns in trend df — trend prediction unavailable.")
            return

        # Scale the time axes
        X_raw    = df[self.time_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_raw)

        skipped = []
        for col in expected_features:
            if col in self.time_cols:
                continue
            if col not in df.columns:
                skipped.append(col)
                continue

            y = df[col]

            # ── Skip all-NaN columns ──────────────────────────────────
            valid_mask = y.notna()
            if valid_mask.sum() == 0:
                skipped.append(col)
                print(f"   ⚠️  Skipping all-NaN column: {col}")
                continue

            # ── NUMERIC — Ridge Regression ────────────────────────────
            if pd.api.types.is_numeric_dtype(y):
                y_clean = y.copy()
                # Fill NaN with median of valid values only
                median_val = y_clean[valid_mask].median()
                y_clean = y_clean.fillna(median_val if pd.notna(median_val) else 0)

                try:
                    model = Ridge(alpha=1.0)
                    model.fit(X_scaled, y_clean)
                    self.models[col] = {'type': 'numeric', 'model': model,
                                        'fallback': float(median_val) if pd.notna(median_val) else 0.0}
                except Exception as e:
                    print(f"   ⚠️  Ridge fit failed for '{col}': {e}")
                    skipped.append(col)

            # ── CATEGORICAL — KNN Classifier ──────────────────────────
            else:
                mode_val = y[valid_mask].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                y_clean  = y.fillna(fill_val).astype(str)

                le        = LabelEncoder()
                y_encoded = le.fit_transform(y_clean)

                # KNN needs at least k neighbours
                n_neighbors = min(5, valid_mask.sum())
                if n_neighbors < 1:
                    skipped.append(col)
                    continue

                try:
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                    model.fit(X_scaled, y_encoded)
                    self.models[col] = {'type': 'categorical', 'model': model,
                                        'le': le, 'fallback': fill_val}
                except Exception as e:
                    print(f"   ⚠️  KNN fit failed for '{col}': {e}")
                    skipped.append(col)

        self.is_trained = True
        if skipped:
            print(f"   [trend] Skipped columns: {skipped}")
        print(f"   [trend] Trained models for {len(self.models)} features.")

    def predict_for_date(self, future_time_index, year, month):
        """
        Returns a single-row DataFrame with predicted feature values.
        Columns match self.expected_features.
        """
        input_data = {}
        if 'time_index' in self.time_cols: input_data['time_index'] = [future_time_index]
        if 'year'       in self.time_cols: input_data['year']       = [year]
        if 'month'      in self.time_cols: input_data['month']      = [month]

        X_future = pd.DataFrame(input_data)

        try:
            X_scaled = self.scaler.transform(X_future)
        except Exception as e:
            print(f"   ⚠️  Scaler transform failed: {e}. Using raw values.")
            X_scaled = X_future.values

        predicted_row = {}

        for col in self.expected_features:
            # Time columns: pass through directly
            if col in self.time_cols:
                predicted_row[col] = X_future[col].iloc[0] if col in X_future else np.nan
                continue

            if col not in self.models:
                # Column was skipped (all-NaN etc.) — use NaN, pipeline imputer handles it
                predicted_row[col] = np.nan
                continue

            info = self.models[col]
            try:
                if info['type'] == 'numeric':
                    predicted_row[col] = round(float(info['model'].predict(X_scaled)[0]), 4)
                else:
                    enc = info['model'].predict(X_scaled)[0]
                    predicted_row[col] = info['le'].inverse_transform([enc])[0]
            except Exception as e:
                print(f"   ⚠️  Predict failed for '{col}': {e}")
                predicted_row[col] = info.get('fallback', np.nan)

        return pd.DataFrame([predicted_row])