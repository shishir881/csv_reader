import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import entropy

# Columns that are always date-derived — never typed by user, never in training_df
DATE_DERIVED_COLS = ['time_index', 'year', 'month']


class FeatureEngineer:
    """
    Splits an incoming DataFrame into two completely separate processed DataFrames:

      training_df  — numeric + categorical columns only (no date cols).
                     This is what the ML model is trained on.
                     This is also what the user fills in at prediction time.

      date_df      — time_index, year, month columns aligned by index.
                     Used ONLY for the trend prediction path.
                     Never mixed into training_df.

    Public attributes after preprocess():
      self.training_df      — ready-to-train DataFrame (includes target)
      self.date_df          — date-derived feature DataFrame (or empty)
      self.reference_date   — earliest date in dataset (for trend prediction)
      self.trend_freq       — 'daily' / 'monthly' / 'yearly'
      self.dropped_leakage  — list of dropped columns
      self.dropped_noise    — list of dropped columns
      self.dropped_redundant— list of dropped columns
    """

    def __init__(self, df, target_col=None, date_col=None, problem_type='regression'):
        self.raw_df       = df.copy()
        self.target_col   = target_col
        self.date_col     = date_col
        self.problem_type = problem_type

        # Outputs
        self.training_df   = pd.DataFrame()
        self.date_df       = pd.DataFrame()

        # Date metadata (for trend prediction)
        self.reference_date = None
        self.trend_freq     = None

        # Dropped column tracking (for diagnostics)
        self.dropped_leakage   = []
        self.dropped_noise     = []
        self.dropped_redundant = []

    @property
    def dropped_cols(self):
        return self.dropped_leakage + self.dropped_noise + self.dropped_redundant

    # ──────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────

    def preprocess(self):
        """
        Main entry point. Populates self.training_df and self.date_df.
        Returns self.training_df for backward compatibility.
        """
        df = self.raw_df.copy()

        # 1. Extract and process the date column separately
        df = self._process_date(df)

        # 2. Clean the target column
        df = self._clean_target(df)

        # 3. Drop all-NaN columns early — before MI scoring or pipeline building
        #    (these cause SimpleImputer warnings and NaN crashes in trend predictor)
        df = self._drop_all_nan_cols(df)

        # 4. Filter features (leakage / noise / redundancy) on num+cat only
        df = self._filter_features(df)

        # 5. Store result — training_df has NO date-derived columns
        self.training_df = df
        print(f"   [preprocess] training_df shape : {self.training_df.shape}")
        print(f"   [preprocess] date_df shape     : {self.date_df.shape}")
        return self.training_df

    # ──────────────────────────────────────────────────────────
    # STEP 1 — DATE PROCESSING (completely isolated)
    # ──────────────────────────────────────────────────────────

    def _process_date(self, df):
        """
        If a date column exists:
          - Parse it, sort by it, compute reference_date and trend_freq
          - Build self.date_df with [time_index, year, month] at the same index
          - Drop the original date column from df
          - Do NOT add time_index/year/month back to df

        Returns df without the date column (and without any date-derived cols).
        """
        if not self.date_col or self.date_col not in df.columns:
            return df

        try:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df = df.sort_values(by=self.date_col).reset_index(drop=True)

            self.reference_date = df[self.date_col].min()
            avg_gap = df[self.date_col].diff().dt.days.median()

            if pd.isna(avg_gap):
                self.trend_freq = 'daily'
                time_index = pd.Series(0, index=df.index)
            elif 25 <= avg_gap <= 35:
                self.trend_freq = 'monthly'
                print("📅 Detected Monthly Trend.")
                time_index = (
                    (df[self.date_col].dt.year - self.reference_date.year) * 12 +
                    (df[self.date_col].dt.month - self.reference_date.month)
                )
            elif avg_gap >= 360:
                self.trend_freq = 'yearly'
                print("📅 Detected Yearly Trend.")
                time_index = df[self.date_col].dt.year - self.reference_date.year
            else:
                self.trend_freq = 'daily'
                print("📅 Detected Daily/Continuous Trend.")
                time_index = (df[self.date_col] - self.reference_date).dt.days

            # Build date_df — separate, never touched by training pipeline
            self.date_df = pd.DataFrame({
                'time_index': time_index.values,
                'year':       df[self.date_col].dt.year.values,
                'month':      df[self.date_col].dt.month.values,
            }, index=df.index)

            print(f"   [_process_date] date_df built: {list(self.date_df.columns)}, freq={self.trend_freq}")

        except Exception as e:
            print(f"   ⚠️  Date processing failed: {e}")

        # Always drop the original date column from the training path
        df = df.drop(columns=[self.date_col], errors='ignore')
        return df

    # ──────────────────────────────────────────────────────────
    # STEP 2 — TARGET CLEANING
    # ──────────────────────────────────────────────────────────

    def _clean_target(self, df):
        if not self.target_col or self.target_col not in df.columns:
            return df

        df = df.dropna(subset=[self.target_col])
        if self.problem_type == 'regression':
            df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
            df = df.dropna(subset=[self.target_col])

        # Also realign date_df to the same surviving rows
        if not self.date_df.empty:
            self.date_df = self.date_df.loc[df.index].reset_index(drop=True)
            df = df.reset_index(drop=True)

        return df

    # ──────────────────────────────────────────────────────────
    # STEP 3 — DROP ALL-NaN COLUMNS
    # ──────────────────────────────────────────────────────────

    def _drop_all_nan_cols(self, df):
        """
        Drop feature columns where every single value is NaN.
        These cause SimpleImputer warnings and NaN crashes in the trend predictor.
        The target column is always preserved.
        """
        feature_cols = [c for c in df.columns if c != self.target_col]
        all_nan = [c for c in feature_cols if df[c].isna().all()]
        if all_nan:
            print(f"   [_drop_all_nan_cols] Dropping all-NaN cols: {all_nan}")
            df = df.drop(columns=all_nan)
        return df

    # ──────────────────────────────────────────────────────────
    # STEP 4 — FEATURE FILTERING (MI + redundancy)
    # ──────────────────────────────────────────────────────────

    def _calculate_normalized_mi(self, df, target):
        if df.empty:
            return pd.Series(dtype=float)
        X = df.drop(columns=[target], errors='ignore')
        y = df[target]
        if X.shape[1] == 0:
            return pd.Series(dtype=float)

        X_filled = X.fillna(-1)
        X_encoded = X_filled.copy()
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))

        discrete_features = X_encoded.dtypes == int

        try:
            if self.problem_type == 'classification':
                y_enc = LabelEncoder().fit_transform(y) if y.dtype == 'object' else y
                mi = mutual_info_classif(X_encoded, y_enc,
                                         discrete_features=discrete_features, random_state=42)
                h = entropy(pd.Series(y_enc).value_counts(normalize=True))
                normed = np.zeros(len(mi)) if h == 0 else mi / h
            else:
                y_num = pd.to_numeric(y, errors='coerce').fillna(0)
                mi = mutual_info_regression(X_encoded, y_num,
                                            discrete_features=discrete_features, random_state=42)
                normed = np.sqrt(1 - np.exp(-2 * mi))

            return pd.Series(np.clip(normed, 0, 1), index=X.columns)

        except Exception as e:
            try:
                print(f"   ⚠️  MI retry after error: {e}")
                y_num = pd.to_numeric(y, errors='coerce').fillna(0)
                mi = mutual_info_regression(X_encoded.astype(float), y_num, random_state=42)
                return pd.Series(np.clip(np.sqrt(1 - np.exp(-2 * mi)), 0, 1), index=X.columns)
            except Exception:
                print("   ⚠️  MI failed entirely — keeping all features.")
                return pd.Series(0.5, index=X.columns)

    def _filter_features(self, df):
        if df.empty or not self.target_col or self.target_col not in df.columns:
            return df

        mi_scores = self._calculate_normalized_mi(df, self.target_col)
        if mi_scores.empty:
            return df

        print("   [MI Scores]")
        print(mi_scores.sort_values(ascending=False).to_string())

        # Leakage: MI > 0.97
        leakage = mi_scores[mi_scores > 0.99].index.tolist()
        if leakage:
            self.dropped_leakage.extend(leakage)

        # Noise: MI == 0.0 exactly
        noise = mi_scores[mi_scores == 0.0].index.tolist()
        if noise:
            self.dropped_noise.extend(noise)

        to_drop = list(set(self.dropped_leakage + self.dropped_noise))
        df = df.drop(columns=to_drop, errors='ignore')

        # Redundancy: high correlation between remaining numeric features
        num_df = df.select_dtypes(include=['float64', 'int64'])
        if num_df.shape[1] > 1:
            corr = num_df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            redundant = [c for c in upper.columns if any(upper[c] > 0.97)]
            if redundant:
                self.dropped_redundant.extend(redundant)
                df = df.drop(columns=redundant, errors='ignore')

        return df

    # ──────────────────────────────────────────────────────────
    # PIPELINE BUILDER (for ModelTrainer)
    # ──────────────────────────────────────────────────────────

    def get_sklearn_pipeline(self, num_cols, cat_cols):
        """
        Build a ColumnTransformer pipeline for the given num/cat columns.
        Only operates on training_df columns — date-derived cols are never included.
        """
        all_dropped = set(self.dropped_cols)
        num_cols = [c for c in num_cols if c not in all_dropped]
        cat_cols = [c for c in cat_cols if c not in all_dropped]

        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
        ])
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot',  OneHotEncoder(handle_unknown='ignore')),
        ])
        return ColumnTransformer([
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
        ])

    # ──────────────────────────────────────────────────────────
    # HELPERS for trend prediction path
    # ──────────────────────────────────────────────────────────

    def get_trend_training_df(self):
        """
        Returns a DataFrame with date_df columns + training_df feature columns
        (excluding target). Used to train the AdvancedTrendPredictor so it can
        learn to predict each feature from the time index.
        """
        if self.date_df.empty or self.training_df.empty:
            return self.training_df.copy()

        feature_cols = [c for c in self.training_df.columns if c != self.target_col]
        return pd.concat(
            [self.date_df.reset_index(drop=True),
             self.training_df[feature_cols].reset_index(drop=True)],
            axis=1
        )