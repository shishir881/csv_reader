# ML Module Integration Guide

## ✅ Fixed Issues

### 1. **Trend Predictor Serialization**

- Added `save()` and `load()` methods to `AdvancedTrendPredictor`
- Added `is_trained` flag to track training status
- Fixed scaler transformation error handling in `predict_for_date()`

### 2. **Missing Trend Predictor Initialization**

- Fixed `BatchPredictor.__init__()` to properly load trend predictor
- Added `_load_trend_predictor()` method for metadata handling
- Populated `self.trend_predictor`, `self.reference_date`, `self.trend_freq`

### 3. **Broken predict_from_date() Method**

- Removed redundant parameters (was expecting `reference_date` and `trend_freq` as arguments)
- Now uses saved metadata from `model_metadata.json`
- Properly scales output and handles exceptions

### 4. **Missing Trend Predictor Setup**

- Added `setup_trend_predictor()` method in `FeatureEngineer`
- Added `get_metadata()` method to return date-related metadata
- Imported `AdvancedTrendPredictor` in preprocessing

### 5. **Model & Predictor Saving Integration**

- Added `save_model_with_trend_predictor()` method in `ModelTrainer`
- Saves three files together:
  - `model_file.pkl` - Main trained model
  - `trend_predictor.pkl` - Trend predictor instance
  - `model_metadata.json` - Reference date, trend frequency, problem type

---

## 📋 Complete Workflow

### Step 1: Data Inspection & Preprocessing

```python
from read.ml.utils import DataInspector
from read.ml.preprocessing import FeatureEngineer

# Load and inspect data
df = pd.read_csv('your_data.csv')
inspector = DataInspector(df, target_col='Sales')

# Detect problem type
problem_type = inspector.detect_problem_type()  # 'regression' or 'classification'

# Get column types
col_types = inspector.get_column_types()  # {num_cols, cat_cols, date_cols}

# Engineer features
engineer = FeatureEngineer(
    df=df,
    target_col='Sales',
    date_col='Date',  # ← Optional, for trend prediction
    problem_type=problem_type
)

# Preprocess data
df_processed = engineer.preprocess()

# Setup trend predictor (if date column exists)
trend_predictor = engineer.setup_trend_predictor(df_processed)
```

### Step 2: Model Training

```python
from read.ml.trainer import ModelTrainer

# Get preprocessing pipeline
preprocessor = engineer.get_sklearn_pipeline(col_types['num_cols'], col_types['cat_cols'])

# Train model with hyperparameter tuning
trainer = ModelTrainer(
    df=df_processed,
    target_col='Sales',
    preprocessor=preprocessor,
    col_types=col_types,
    problem_type=problem_type
)

results = trainer.train()
```

### Step 3: Model & Trend Predictor Saving

```python
# Save everything together
model_path = 'models/sales_model.pkl'
trainer.save_model_with_trend_predictor(results, engineer, model_path)

# This creates:
# ├── models/sales_model.pkl
# ├── models/trend_predictor.pkl
# └── models/model_metadata.json
```

### Step 4: Batch Prediction

```python
from read.ml.predictor import BatchPredictor

# Load model (automatically loads trend predictor + metadata)
predictor = BatchPredictor('models/sales_model.pkl')

# Predict on new data
df_new = pd.read_csv('new_data.csv')
predictions_df, status = predictor.predict(df_new)
```

### Step 5: Time-Series Prediction (Only if Date Column Exists)

```python
# Predict using ONLY a date
output_df, status = predictor.predict_from_date('2025-06-15')

if status == "Success":
    print(output_df)
    # Shows predicted values for all features on that date
else:
    print(f"Error: {status}")
```

---

## 📁 Model Directory Structure

After training and saving:

```
models/
├── sales_model.pkl
│   └── Main trained pipeline (sklearn.pipeline.Pipeline)
├── trend_predictor.pkl
│   └── AdvancedTrendPredictor instance
│       ├── models (dict of Ridge/KNN models)
│       ├── scaler (StandardScaler)
│       ├── expected_features (list)
│       └── time_cols (list)
└── model_metadata.json
    ├── reference_date (first date in dataset)
    ├── trend_freq ("daily"/"monthly"/"yearly")
    ├── has_trend_predictor (boolean)
    ├── problem_type ("regression"/"classification")
    └── best_model_name ("Ridge Regression", etc.)
```

---

## 🔧 Key Classes & Methods

### FeatureEngineer

```python
engineer = FeatureEngineer(df, target_col, date_col, problem_type)

engineer.preprocess()  # Cleans data, creates time features
engineer.setup_trend_predictor(df_processed)  # Initializes & trains trend predictor
engineer.get_metadata()  # Returns {reference_date, trend_freq, has_trend_predictor}
engineer.get_sklearn_pipeline(num_cols, cat_cols)  # Returns preprocessing pipeline
```

### ModelTrainer

```python
trainer = ModelTrainer(df, target_col, preprocessor, col_types, problem_type)

results = trainer.train()  # Returns best model + metrics
trainer.save_model_with_trend_predictor(results, engineer, model_path)  # Saves all 3 files
```

### AdvancedTrendPredictor

```python
predictor = AdvancedTrendPredictor()

predictor.train_trends(df, expected_features)  # Trains mini-models for each feature
predictor.predict_for_date(future_time_index, year, month)  # Returns predicted feature DataFrame
predictor.save(filepath)  # Serializes to joblib
predictor.load(filepath)  # Deserializes from joblib (static method)
```

### BatchPredictor

```python
predictor = BatchPredictor(model_path)  # Auto-loads model + trend predictor

# Standard batch prediction
predictions_df, status = predictor.predict(input_data)
# input_data can be file path (str) or DataFrame

# Time-series prediction (if trend predictor available)
output_df, status = predictor.predict_from_date('2025-06-15')
```

---

## ⚠️ Important Notes

1. **Trend Predictor Only Works With Date Columns**: If your dataset doesn't have a date column, `trend_predictor` will be `None`.

2. **Date Column Requirement**: The date column must be detectable by pandas (standard formats like "2025-01-15", "2025-01-15 10:30:00", etc.).

3. **Frequency Detection**: Automatic frequency detection based on date gaps:
   - **25-35 days**: Monthly trend
   - **≥360 days**: Yearly trend
   - **< 25 days**: Daily/continuous trend

4. **Time Features Created**:
   - `time_index`: Sequence number from first date
   - `year`: Year of date
   - `month`: Month of date

5. **Original Date Column**: Removed after preprocessing (sklearn doesn't handle datetime directly).

---

## 🐛 Debugging

### Missing Trend Predictor

```python
predictor = BatchPredictor('models/sales_model.pkl')
if predictor.trend_predictor is None:
    print("No trend predictor available - model was trained without date column")
```

### Check Metadata

```python
import json
with open('models/model_metadata.json') as f:
    metadata = json.load(f)
    print(metadata)
```

### Verify Training Status

```python
if not predictor.trend_predictor.is_trained:
    print("Trend predictor data corrupted - was not properly trained")
```

---

## 📊 Example: Complete End-to-End Usage

```python
import pandas as pd
from read.ml.utils import DataInspector
from read.ml.preprocessing import FeatureEngineer
from read.ml.trainer import ModelTrainer
from read.ml.predictor import BatchPredictor

# 1. LOAD & INSPECT
df = pd.read_csv('sales_data.csv')
inspector = DataInspector(df, 'Revenue')
is_valid, msg = inspector.sanity_check()
print(msg)

problem_type = inspector.detect_problem_type()
col_types = inspector.get_column_types()

# 2. PREPROCESS
engineer = FeatureEngineer(df, 'Revenue', 'Date', problem_type)
df_processed = engineer.preprocess()
engineer.setup_trend_predictor(df_processed)

# 3. TRAIN
preprocessor = engineer.get_sklearn_pipeline(col_types['num_cols'], col_types['cat_cols'])
trainer = ModelTrainer(df_processed, 'Revenue', preprocessor, col_types, problem_type)
results = trainer.train()

# 4. SAVE
trainer.save_model_with_trend_predictor(results, engineer, 'models/revenue_model.pkl')

# 5. PREDICT
predictor = BatchPredictor('models/revenue_model.pkl')

# Batch prediction
new_df = pd.read_csv('future_data.csv')
preds, status = predictor.predict(new_df)

# Date-based prediction
date_preds, status = predictor.predict_from_date('2026-06-15')
print(date_preds)
```

---

## ✨ What's Fixed Now

| Issue                             | Solution                                       |
| --------------------------------- | ---------------------------------------------- |
| Trend predictor not loading       | Added `_load_trend_predictor()`                |
| Missing metadata                  | Added JSON metadata file saving                |
| Broken `predict_from_date()`      | Now uses saved `reference_date` & `trend_freq` |
| Trend predictor not serializable  | Added `save()` and `load()` methods            |
| Scaler transform error            | Added try-catch with fallback                  |
| No trend predictor initialization | Added `setup_trend_predictor()` method         |
| Model & predictor not bundled     | Added `save_model_with_trend_predictor()`      |

All issues have been resolved! ✅
