import json
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .forms import CustomSignupForm, CustomLoginForm, DatasetForm, PredictionForm
from .models import Dataset
import os
import joblib

from read.ml.utils import DataInspector
from read.ml.preprocessing import FeatureEngineer, DATE_DERIVED_COLS
from read.ml.trainer import ModelTrainer
from read.ml.diagnostics import ModelDiagnoser
from django.conf import settings
from read.ml.predictor import BatchPredictor
from read.ml.visualizer import DataVisualizer
from read.ml.trend_predictor import AdvancedTrendPredictor, get_future_time_index


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _get_model_paths(dataset_id):
    """Return (model_path, meta_path) for a given dataset_id."""
    model_dir  = os.path.join(settings.MEDIA_ROOT, 'models')
    model_path = os.path.join(model_dir, f'model_{dataset_id}.pkl')
    meta_path  = os.path.join(model_dir, f'model_{dataset_id}_meta.json')
    return model_path, meta_path


def _extract_pipeline_features(pipeline):
    """
    Pull the exact column lists the ColumnTransformer was fitted on.
    Returns (num_cols, cat_cols, all_feature_cols).
    """
    preprocessor = pipeline.named_steps['preprocessor']
    num_cols = list(preprocessor.transformers_[0][2])  # 'num' transformer
    cat_cols = list(preprocessor.transformers_[1][2])  # 'cat' transformer
    return num_cols, cat_cols, num_cols + cat_cols


def _compute_date_features(date_str, original_df, date_col):
    """
    Given a date string and the original DataFrame, compute
    time_index, year, month using the same logic as FeatureEngineer.
    Returns a dict of {col_name: value} for whichever date cols exist.
    """
    tmp = original_df[[date_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.sort_values(date_col)
    reference_date = tmp[date_col].min()
    avg_gap = tmp[date_col].diff().dt.days.median()

    if pd.isna(avg_gap):
        trend_freq = 'daily'
    elif 25 <= avg_gap <= 35:
        trend_freq = 'monthly'
    elif avg_gap >= 360:
        trend_freq = 'yearly'
    else:
        trend_freq = 'daily'

    future_index, year, month = get_future_time_index(date_str, reference_date, trend_freq)
    return {
        'time_index': float(future_index),
        'year':       float(year),
        'month':      float(month),
    }


# ──────────────────────────────────────────────────────────────
# AUTH
# ──────────────────────────────────────────────────────────────

def signup_view(request):
    if request.method == 'POST':
        form = CustomSignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('upload')
    else:
        form = CustomSignupForm()
    return render(request, 'registration/signup.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = CustomLoginForm(data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect(request.POST.get('next', 'upload'))
    else:
        form = CustomLoginForm()
    return render(request, 'registration/login.html', {'form': form})


def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return redirect('login')


# ──────────────────────────────────────────────────────────────
# UPLOAD / SELECT TARGET
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def upload_view(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save()
            return redirect('select_target', dataset_id=dataset.id)
    else:
        form = DatasetForm()
    return render(request, 'dashboard/upload.html', {'form': form})


@login_required(login_url='login')
def select_target_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    try:
        df      = pd.read_csv(dataset.file.path)
        columns = df.columns.tolist()
    except Exception as e:
        return render(request, 'dashboard/error.html', {'error': f"CSV read error: {e}"})

    if request.method == 'POST':
        target = request.POST.get('target')
        return train_model_view(request, dataset_id, target)

    return render(request, 'dashboard/select_target.html', {
        'columns': columns,
        'dataset': dataset,
    })


# ──────────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def train_model_view(request, dataset_id, target_col):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    df      = pd.read_csv(dataset.file.path)

    # Keep a clean copy of original column names BEFORE any preprocessing
    original_csv_cols = df.columns.tolist()

    print(f"\n{'='*55}")
    print(f"  ML Pipeline | {dataset.name} | target={target_col}")
    print(f"{'='*55}")

    try:
        # ── 1. Inspect ───────────────────────────────────────
        inspector = DataInspector(df, target_col)
        is_valid, msg = inspector.sanity_check()
        if not is_valid:
            return render(request, 'dashboard/error.html', {'error': msg})

        problem_type = inspector.detect_problem_type()
        col_types    = inspector.get_column_types()
        date_col     = col_types['date_cols'][0] if col_types['date_cols'] else None
        print(f"  problem={problem_type}  date_col={date_col}")

        # ── 2. Preprocess ────────────────────────────────────
        engineer = FeatureEngineer(
            df, target_col, date_col=date_col, problem_type=problem_type
        )
        clean_df = engineer.preprocess()

        print(f"  clean shape={clean_df.shape}")
        print(f"  date_derived_cols={engineer.date_derived_cols}")
        print(f"  dropped leakage={engineer.dropped_leakage}")
        print(f"  dropped noise={engineer.dropped_noise}")
        print(f"  dropped redundant={engineer.dropped_redundant}")

        # ── 3. Train ─────────────────────────────────────────
        clean_inspector  = DataInspector(clean_df, target_col)
        clean_col_types  = clean_inspector.get_column_types()
        pipeline = engineer.get_sklearn_pipeline(
            clean_col_types['num_cols'], clean_col_types['cat_cols']
        )
        trainer = ModelTrainer(
            clean_df, target_col, pipeline, clean_col_types, problem_type=problem_type
        )
        results = trainer.train()

        best_name    = results['best_model_name']
        best_metrics = results['metrics'][best_name]
        best_model   = results['best_model']
        print(f"  winner={best_name}  metrics={best_metrics}")

        # ── 4. Extract EXACT feature list from fitted pipeline ─
        #
        # This is the ground truth: the ColumnTransformer records
        # exactly which columns it was fitted on.
        num_cols, cat_cols, model_feature_cols = _extract_pipeline_features(best_model)
        print(f"  model_feature_cols={model_feature_cols}")

        # ── 5. Decide what the user should input ──────────────
        #
        # Rule: show a column in the form iff it is:
        #   (a) actually used by the model  AND
        #   (b) present in the original CSV  (meaning the user can know its value)
        #
        # Columns NOT in original CSV = date-derived ones (time_index, year, month)
        # Those will be auto-computed from a date picker when needed.
        #
        user_input_cols = [
            c for c in model_feature_cols
            if c in original_csv_cols          # came from the CSV
            and c != date_col                  # not the raw date string col
            and c != target_col               # not the target
        ]

        # Date-derived cols the model needs but the user never types
        auto_date_cols = [
            c for c in model_feature_cols
            if c in DATE_DERIVED_COLS
        ]

        needs_date_input = len(auto_date_cols) > 0
        print(f"  user_input_cols={user_input_cols}")
        print(f"  auto_date_cols={auto_date_cols}")

        # ── 6. Visualize ─────────────────────────────────────
        try:
            plots = DataVisualizer(clean_df, target_col, problem_type).generate_all()
        except Exception as e:
            print(f"  viz failed: {e}")
            plots = {}

        # ── 7. Diagnose ──────────────────────────────────────
        diagnosis = ModelDiagnoser(results, engineer, target_col).get_diagnosis()

        # ── 8. Save model + metadata ─────────────────────────
        model_path, meta_path = _get_model_paths(dataset_id)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)

        # Metadata lets predict_view reconstruct exactly what the model needs
        meta = {
            'target_col':        target_col,
            'date_col':          date_col,
            'original_csv_cols': original_csv_cols,
            'model_feature_cols': model_feature_cols,
            'num_cols':          num_cols,
            'cat_cols':          cat_cols,
            'user_input_cols':   user_input_cols,
            'auto_date_cols':    auto_date_cols,
            'needs_date_input':  needs_date_input,
            'problem_type':      problem_type,
            'file_path':         dataset.file.path,
            'model_path':        model_path,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  metadata saved → {meta_path}")

        # ── 9. Build context for template ────────────────────
        # Feature types for form rendering
        feature_types = {}
        for col in user_input_cols:
            feature_types[col] = (
                'number' if pd.api.types.is_numeric_dtype(clean_df[col]) else 'text'
            )

        # Sample rows — only user-visible columns
        reference_data = clean_df[user_input_cols].head(5).to_dict(orient='records')

        has_trend = date_col is not None and engineer.reference_date is not None

        context = {
            'results':      results,
            'best_metrics': best_metrics,
            'diagnosis':    diagnosis,
            'target':       target_col,
            'problem_type': problem_type,
            'dataset_id':   dataset_id,
            # Prediction form
            'feature_columns':   user_input_cols,
            'feature_types':     feature_types,
            'reference_data':    reference_data,
            'needs_date_input':  needs_date_input,
            'auto_date_cols':    auto_date_cols,
            # Trend forecast
            'has_trend':  has_trend,
            'date_col':   date_col or '',
            'target_col': target_col,
            'file_path':  dataset.file.path,
            'model_path': model_path,
            # Charts
            'plots': plots,
        }
        return render(request, 'dashboard/result.html', context)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, 'dashboard/error.html', {
            'error': f"ML Pipeline Failed: {str(e)}"
        })


# ──────────────────────────────────────────────────────────────
# PREDICT (manual feature entry)
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def predict_view(request, dataset_id):
    """
    Prediction via manual feature entry.

    Flow:
      1. Load saved metadata (exact feature list, types, etc.)
      2. Collect user-typed values for user_input_cols only
      3. If model needs date features, compute them from the date picker
      4. Build a single-row DataFrame with ALL model_feature_cols
      5. Run model.predict() and render result
    """
    if request.method != 'POST':
        return redirect('upload')

    model_path, meta_path = _get_model_paths(dataset_id)

    # ── Guard: model must exist ───────────────────────────────
    if not os.path.exists(model_path):
        return render(request, 'dashboard/error.html', {
            'error': "Trained model not found. Please re-train the model first."
        })
    if not os.path.exists(meta_path):
        return render(request, 'dashboard/error.html', {
            'error': "Model metadata not found. Please re-train the model."
        })

    try:
        # ── Load metadata & model ─────────────────────────────
        with open(meta_path) as f:
            meta = json.load(f)

        model            = joblib.load(model_path)
        target_col       = meta['target_col']
        date_col         = meta['date_col']
        user_input_cols  = meta['user_input_cols']
        auto_date_cols   = meta['auto_date_cols']
        model_feature_cols = meta['model_feature_cols']
        original_csv_cols  = meta['original_csv_cols']

        dataset     = Dataset.objects.get(id=dataset_id)
        original_df = pd.read_csv(dataset.file.path)

        # ── Collect user-typed values ─────────────────────────
        # Only accept keys that are in user_input_cols — nothing else.
        input_data = {}
        for col in user_input_cols:
            raw = request.POST.get(col, '').strip()
            if pd.api.types.is_numeric_dtype(original_df[col]):
                try:
                    input_data[col] = float(raw)
                except (ValueError, TypeError):
                    input_data[col] = float(original_df[col].median())
            else:
                input_data[col] = raw if raw else str(original_df[col].mode()[0])

        # ── Auto-compute date-derived features ────────────────
        if auto_date_cols and date_col:
            date_str = request.POST.get('predict_date', '').strip()
            if not date_str:
                # Default: use today
                date_str = pd.Timestamp.today().strftime('%Y-%m-%d')

            date_features = _compute_date_features(date_str, original_df, date_col)

            # Only add the ones the model actually uses
            for col in auto_date_cols:
                if col in date_features:
                    input_data[col] = date_features[col]

        # ── Build the full feature row in correct column order ─
        # model_feature_cols is the exact order the pipeline expects
        row = {col: input_data.get(col, np.nan) for col in model_feature_cols}
        input_df = pd.DataFrame([row])

        print(f"\n[predict_view] input_df columns : {list(input_df.columns)}")
        print(f"[predict_view] input_df values  : {input_df.iloc[0].to_dict()}")

        # ── Predict ───────────────────────────────────────────
        prediction = model.predict(input_df)[0]
        if isinstance(prediction, (int, float, np.integer, np.floating)):
            prediction = round(float(prediction), 2)

        # Show user only the values they typed (hide auto date cols)
        display_inputs = {k: v for k, v in input_data.items()
                          if k in user_input_cols}

        return render(request, 'dashboard/prediction_result.html', {
            'inputs':     display_inputs,
            'prediction': prediction,
            'dataset_id': dataset_id,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, 'dashboard/error.html', {
            'error': f"Prediction failed: {str(e)}"
        })


# ──────────────────────────────────────────────────────────────
# TREND FORECAST (date-only, all features auto-predicted)
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def predict_future_trend(request):
    if request.method != 'POST':
        return render(request, 'dashboard/error.html', {
            'error': 'Invalid request method.'
        })

    try:
        file_path       = request.POST.get('file_path', '').strip()
        model_path      = request.POST.get('model_path', '').strip()
        future_date_str = request.POST.get('future_date', '').strip()
        date_col        = request.POST.get('date_col', '').strip()
        target_col      = request.POST.get('target_col', '').strip()

        missing = [k for k, v in {
            'file_path': file_path, 'model_path': model_path,
            'future_date': future_date_str,
            'date_col': date_col, 'target_col': target_col,
        }.items() if not v]

        if missing:
            return render(request, 'dashboard/error.html', {
                'error': f"Missing fields: {', '.join(missing)}"
            })

        for path, label in [(model_path, 'model'), (file_path, 'dataset')]:
            if not os.path.exists(path):
                return render(request, 'dashboard/error.html', {
                    'error': f"File not found on server: {label}"
                })

        main_model  = joblib.load(model_path)
        original_df = pd.read_csv(file_path)

        engineer     = FeatureEngineer(original_df, target_col=target_col, date_col=date_col)
        processed_df = engineer.preprocess()

        if engineer.reference_date is None:
            return render(request, 'dashboard/error.html', {
                'error': f"Could not detect date sequence in column '{date_col}'."
            })

        # Exact feature cols the model was trained on
        _, _, expected_features = _extract_pipeline_features(main_model)

        trend_predictor = AdvancedTrendPredictor()
        trend_predictor.train_trends(processed_df, expected_features)

        future_index, year, month = get_future_time_index(
            future_date_str, engineer.reference_date, engineer.trend_freq
        )
        predicted_df = trend_predictor.predict_for_date(future_index, year, month)

        final_prediction = main_model.predict(predicted_df)[0]
        if isinstance(final_prediction, (float, np.floating)):
            final_prediction = round(float(final_prediction), 2)

        # Display: hide internal date cols
        features_dict = {
            k: v for k, v in predicted_df.iloc[0].to_dict().items()
            if k not in DATE_DERIVED_COLS
        }

        return render(request, 'dashboard/trend_result.html', {
            'future_date':       future_date_str,
            'final_prediction':  final_prediction,
            'target_col':        target_col,
            'predicted_features': features_dict,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, 'dashboard/error.html', {
            'error': f"Trend prediction failed: {str(e)}"
        })