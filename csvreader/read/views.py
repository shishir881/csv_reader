import json
import hashlib
import glob
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

def _model_paths(dataset_id):
    d = os.path.join(settings.MEDIA_ROOT, 'models')
    return (
        os.path.join(d, f'model_{dataset_id}.pkl'),
        os.path.join(d, f'model_{dataset_id}_meta.json'),
    )


def _compute_data_fingerprint(file_path, target_col):
    """
    Create a deterministic fingerprint from CSV content + target column.
    Same data + same target → same fingerprint → same trained model
    (because all random_state values are fixed at 42).
    """
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    h.update(target_col.encode('utf-8'))
    return h.hexdigest()


def _find_existing_model_by_fingerprint(fingerprint):
    """
    Scan all existing metadata files for a matching fingerprint.
    Returns the model_path of the existing model if found, else None.
    """
    models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
    if not os.path.isdir(models_dir):
        return None

    for meta_file in glob.glob(os.path.join(models_dir, '*_meta.json')):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get('data_fingerprint') == fingerprint:
                existing_path = meta.get('model_path', '')
                if os.path.exists(existing_path):
                    return existing_path
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return None


def _extract_pipeline_features(pipeline):
    """Pull exact num_cols + cat_cols from the fitted ColumnTransformer."""
    pre = pipeline.named_steps['preprocessor']
    num = list(pre.transformers_[0][2])
    cat = list(pre.transformers_[1][2])
    return num, cat, num + cat


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
        return train_model_view(request, dataset_id, request.POST.get('target'))

    return render(request, 'dashboard/select_target.html',
                  {'columns': columns, 'dataset': dataset})


# ──────────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def train_model_view(request, dataset_id, target_col):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    df      = pd.read_csv(dataset.file.path)
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

        # ── 2. Preprocess ─────────────────────────────────────
        # New architecture:
        #   engineer.training_df  → num+cat only, no date cols → model training
        #   engineer.date_df      → time_index/year/month     → trend prediction only
        engineer = FeatureEngineer(
            df, target_col, date_col=date_col, problem_type=problem_type
        )
        training_df = engineer.preprocess()  # returns engineer.training_df

        print(f"  training_df cols : {list(training_df.columns)}")
        print(f"  date_df cols     : {list(engineer.date_df.columns)}")
        print(f"  dropped leakage  : {engineer.dropped_leakage}")
        print(f"  dropped noise    : {engineer.dropped_noise}")
        print(f"  dropped redundant: {engineer.dropped_redundant}")

        # ── 3. Train on training_df only ──────────────────────
        # Re-detect col types on the clean training df
        clean_inspector = DataInspector(training_df, target_col)
        clean_col_types = clean_inspector.get_column_types()
        pipeline = engineer.get_sklearn_pipeline(
            clean_col_types['num_cols'], clean_col_types['cat_cols']
        )
        trainer = ModelTrainer(
            training_df, target_col, pipeline, clean_col_types, problem_type=problem_type
        )
        results      = trainer.train()
        best_name    = results['best_model_name']
        best_metrics = results['metrics'][best_name]
        best_model   = results['best_model']
        print(f"  winner={best_name}  metrics={best_metrics}")

        # ── 4. Extract EXACT feature list from fitted pipeline ─
        # This is the ground truth: only these columns exist in the model.
        # Since training_df has NO date cols, model_feature_cols will NEVER
        # include time_index / year / month — by design.
        num_cols, cat_cols, model_feature_cols = _extract_pipeline_features(best_model)
        print(f"  model_feature_cols={model_feature_cols}")

        # ── 5. Determine what user should fill in ─────────────
        # user_input_cols = model_feature_cols that came from the original CSV
        # (all of them, since date cols were never added to training_df)
        user_input_cols = [
            c for c in model_feature_cols
            if c in original_csv_cols and c != target_col
        ]
        print(f"  user_input_cols={user_input_cols}")

        # ── 6. Visualize ─────────────────────────────────────
        try:
            plots = DataVisualizer(training_df, target_col, problem_type).generate_all()
        except Exception as e:
            print(f"  viz failed: {e}")
            plots = {}

        # ── 7. Diagnose ──────────────────────────────────────
        diagnosis = ModelDiagnoser(results, engineer, target_col).get_diagnosis()

        # ── 8. Save model + metadata ─────────────────────────
        model_path, meta_path = _model_paths(dataset_id)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Deduplicate: compute fingerprint and check for existing identical model
        fingerprint = _compute_data_fingerprint(dataset.file.path, target_col)
        existing_model_path = _find_existing_model_by_fingerprint(fingerprint)

        if existing_model_path and existing_model_path != model_path:
            # Identical model already exists — reuse it, skip saving duplicate
            model_path = existing_model_path
            print(f"  ♻️  Reusing existing model → {existing_model_path}")
            print(f"     (same data + same target = identical model, skipping save)")
        else:
            # No duplicate found — save the new model
            joblib.dump(best_model, model_path)
            print(f"  model saved → {model_path}")

        meta = {
            'target_col':          target_col,
            'date_col':            date_col,
            'original_csv_cols':   original_csv_cols,
            'model_feature_cols':  model_feature_cols,
            'num_cols':            num_cols,
            'cat_cols':            cat_cols,
            # user fills in ALL of these — no hidden date cols, no confusion
            'user_input_cols':     user_input_cols,
            'problem_type':        problem_type,
            'file_path':           dataset.file.path,
            'model_path':          model_path,
            # Trend prediction metadata
            'has_trend':           date_col is not None and engineer.reference_date is not None,
            'trend_freq':          engineer.trend_freq,
            'reference_date':      str(engineer.reference_date) if engineer.reference_date else None,
            # Fingerprint for deduplication
            'data_fingerprint':    fingerprint,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  metadata saved → {meta_path}")

        # ── 9. Build context ──────────────────────────────────
        feature_types = {
            col: ('number' if pd.api.types.is_numeric_dtype(training_df[col]) else 'text')
            for col in user_input_cols
        }
        # Sample rows shown as reference in the prediction form
        reference_data = training_df[user_input_cols].head(5).to_dict(orient='records')

        context = {
            'results':      results,
            'best_metrics': best_metrics,
            'diagnosis':    diagnosis,
            'target':       target_col,
            'problem_type': problem_type,
            'dataset_id':   dataset_id,
            # Prediction form — ONLY original CSV columns the model was trained on
            'feature_columns': user_input_cols,
            'feature_types':   feature_types,
            'reference_data':  reference_data,
            # Trend forecast section
            'has_trend':   meta['has_trend'],
            'date_col':    date_col or '',
            'target_col':  target_col,
            'file_path':   dataset.file.path,
            'model_path':  model_path,
            'plots':       plots,
        }
        return render(request, 'dashboard/result.html', context)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, 'dashboard/error.html',
                      {'error': f"ML Pipeline Failed: {str(e)}"})


# ──────────────────────────────────────────────────────────────
# PREDICT (manual feature entry — no date cols involved)
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def predict_view(request, dataset_id):
    """
    Standard prediction: user fills in ONLY the columns the model was
    trained on (all from the original CSV, no date-derived cols).

    Flow:
      1. Load metadata → get exact user_input_cols and model_feature_cols
      2. Parse user POST values for each user_input_col
      3. Build input_df with model_feature_cols in the correct order
      4. model.predict(input_df)
    """
    if request.method != 'POST':
        return redirect('upload')

    model_path, meta_path = _model_paths(dataset_id)

    if not os.path.exists(model_path):
        return render(request, 'dashboard/error.html',
                      {'error': "Model not found. Please re-train."})
    if not os.path.exists(meta_path):
        return render(request, 'dashboard/error.html',
                      {'error': "Model metadata missing. Please re-train."})

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        model              = joblib.load(model_path)
        user_input_cols    = meta['user_input_cols']
        model_feature_cols = meta['model_feature_cols']

        dataset     = Dataset.objects.get(id=dataset_id)
        original_df = pd.read_csv(dataset.file.path)

        # ── Parse only the columns the user is supposed to fill ──
        input_data = {}
        for col in user_input_cols:
            raw = request.POST.get(col, '').strip()
            if col in original_df.columns and pd.api.types.is_numeric_dtype(original_df[col]):
                try:
                    input_data[col] = float(raw)
                except (ValueError, TypeError):
                    # Fallback to column median if user left blank or typed garbage
                    input_data[col] = float(original_df[col].median())
            else:
                input_data[col] = raw if raw else (
                    str(original_df[col].mode().iloc[0]) if col in original_df.columns else ''
                )

        # ── Build full feature row in the exact order the pipeline expects ──
        # model_feature_cols == user_input_cols here (no date cols in model)
        # but we use model_feature_cols to guarantee correct column order
        row      = {col: input_data.get(col, np.nan) for col in model_feature_cols}
        input_df = pd.DataFrame([row])

        print(f"\n[predict_view] input:\n{input_df.to_dict(orient='records')}")

        # ── Predict ───────────────────────────────────────────
        prediction = model.predict(input_df)[0]
        if isinstance(prediction, (int, float, np.integer, np.floating)):
            prediction = round(float(prediction), 2)

        return render(request, 'dashboard/prediction_result.html', {
            'inputs':     input_data,
            'prediction': prediction,
            'dataset_id': dataset_id,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, 'dashboard/error.html',
                      {'error': f"Prediction failed: {str(e)}"})


# ──────────────────────────────────────────────────────────────
# TREND FORECAST (date only — completely separate path)
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def predict_future_trend(request):
    """
    Trend-based prediction:
      1. User picks a future date
      2. AdvancedTrendPredictor (trained on date_df + training features)
         predicts all model feature values for that date
      3. Main model predicts target from those predicted feature values

    Completely separate from predict_view — no user feature entry needed.
    """
    if request.method != 'POST':
        return render(request, 'dashboard/error.html', {'error': 'Invalid request method.'})

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
            return render(request, 'dashboard/error.html',
                          {'error': f"Missing fields: {', '.join(missing)}"})

        for path, label in [(model_path, 'model'), (file_path, 'dataset')]:
            if not os.path.exists(path):
                return render(request, 'dashboard/error.html',
                              {'error': f"File not found on server ({label})."})

        # ── Load model ────────────────────────────────────────
        main_model = joblib.load(model_path)

        # ── Re-run preprocessing to get BOTH dataframes ───────
        # This is the key: we get the exact same training_df and date_df
        # that were used during training.
        original_df = pd.read_csv(file_path)
        engineer    = FeatureEngineer(
            original_df, target_col=target_col, date_col=date_col
        )
        engineer.preprocess()  # populates engineer.training_df and engineer.date_df

        if engineer.reference_date is None:
            return render(request, 'dashboard/error.html', {
                'error': f"Could not detect a date sequence in column '{date_col}'."
            })

        # ── Get exact feature list model was trained on ───────
        _, _, model_feature_cols = _extract_pipeline_features(main_model)

        # ── Build the combined trend training df ──────────────
        # date_df (time_index, year, month) + feature cols side-by-side
        # This teaches the trend predictor: "given this time index → these feature values"
        trend_train_df = engineer.get_trend_training_df()
        print(f"  [trend] trend_train_df shape: {trend_train_df.shape}")
        print(f"  [trend] trend_train_df cols : {list(trend_train_df.columns)}")

        # ── Train the trend predictor ─────────────────────────
        trend_predictor = AdvancedTrendPredictor()
        trend_predictor.train_trends(trend_train_df, model_feature_cols)

        # ── Convert future date → time index ──────────────────
        future_index, year, month = get_future_time_index(
            future_date_str, engineer.reference_date, engineer.trend_freq
        )

        # ── Predict all feature values for that date ──────────
        predicted_features_df = trend_predictor.predict_for_date(future_index, year, month)
        print(f"  [trend] predicted features:\n{predicted_features_df.to_dict(orient='records')}")

        # ── Final prediction from main model ──────────────────
        final_prediction = main_model.predict(predicted_features_df)[0]
        if isinstance(final_prediction, (float, np.floating)):
            final_prediction = round(float(final_prediction), 2)

        # Display: hide date-derived cols from the result page
        features_dict = {
            k: v for k, v in predicted_features_df.iloc[0].to_dict().items()
            if k not in DATE_DERIVED_COLS
        }

        return render(request, 'dashboard/trend_result.html', {
            'future_date':        future_date_str,
            'final_prediction':   final_prediction,
            'target_col':         target_col,
            'predicted_features': features_dict,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, 'dashboard/error.html',
                      {'error': f"Trend prediction failed: {str(e)}"})