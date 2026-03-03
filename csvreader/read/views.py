import json
import hashlib
import glob
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseNotAllowed
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
    Same data + same target -> same fingerprint -> same trained model.
    """
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
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


def _get_user_dataset(request, dataset_id):
    """Get a dataset ensuring it belongs to the current user."""
    return get_object_or_404(Dataset, id=dataset_id, user=request.user)


# ──────────────────────────────────────────────────────────────
# LANDING PAGE (public)
# ──────────────────────────────────────────────────────────────

def landing_view(request):
    """Public landing page. Redirects logged-in users to dashboard."""
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'landing.html')


# ──────────────────────────────────────────────────────────────
# AUTH
# ──────────────────────────────────────────────────────────────

def signup_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = CustomSignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else:
        form = CustomSignupForm()
    return render(request, 'registration/signup.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = CustomLoginForm(data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect(request.POST.get('next', 'dashboard'))
    else:
        form = CustomLoginForm()
    return render(request, 'registration/login.html', {'form': form})


def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return redirect('landing')
    return redirect('landing')


# ──────────────────────────────────────────────────────────────
# DASHBOARD (list of user's datasets)
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def dashboard_view(request):
    """Shows user's uploaded datasets and their status."""
    datasets = Dataset.objects.filter(user=request.user)
    
    # Check which datasets have trained models
    dataset_info = []
    for ds in datasets:
        _, meta_path = _model_paths(ds.id)
        has_model = os.path.exists(meta_path)
        meta = {}
        if has_model:
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                has_model = False
        
        dataset_info.append({
            'dataset': ds,
            'has_model': has_model,
            'target_col': meta.get('target_col', ''),
            'problem_type': meta.get('problem_type', ''),
        })

    return render(request, 'dashboard/home.html', {
        'dataset_info': dataset_info,
    })


# ──────────────────────────────────────────────────────────────
# UPLOAD
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def upload_view(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.user = request.user
            dataset.save()
            return redirect('select_target', dataset_id=dataset.id)
    else:
        form = DatasetForm()
    return render(request, 'dashboard/upload.html', {'form': form})


# ──────────────────────────────────────────────────────────────
# DELETE DATASET
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def delete_dataset_view(request, dataset_id):
    if request.method == 'POST':
        dataset = _get_user_dataset(request, dataset_id)
        # Clean up model files
        model_path, meta_path = _model_paths(dataset_id)
        for path in [model_path, meta_path]:
            if os.path.exists(path):
                os.remove(path)
        # Clean up uploaded file
        if dataset.file and os.path.exists(dataset.file.path):
            os.remove(dataset.file.path)
        dataset.delete()
    return redirect('dashboard')


# ──────────────────────────────────────────────────────────────
# SELECT TARGET
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def select_target_view(request, dataset_id):
    dataset = _get_user_dataset(request, dataset_id)
    try:
        df      = pd.read_csv(dataset.file.path)
        columns = df.columns.tolist()
    except Exception as e:
        return render(request, 'dashboard/error.html', {
            'error': f"CSV read error: {e}",
            'back_url': 'dashboard',
        })

    if request.method == 'POST':
        return train_model_view(request, dataset_id, request.POST.get('target'))

    return render(request, 'dashboard/select_target.html',
                  {'columns': columns, 'dataset': dataset})


# ──────────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def train_model_view(request, dataset_id, target_col=None):
    dataset = _get_user_dataset(request, dataset_id)
    
    # If called directly via URL (e.g., browser refresh on result page),
    # load existing results from metadata
    if target_col is None:
        _, meta_path = _model_paths(dataset_id)
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                target_col = meta.get('target_col')
            except (json.JSONDecodeError, OSError):
                pass
        if not target_col:
            return redirect('select_target', dataset_id=dataset_id)

    df = pd.read_csv(dataset.file.path)
    original_csv_cols = df.columns.tolist()

    print(f"\n{'='*55}")
    print(f"  ML Pipeline | {dataset.name} | target={target_col}")
    print(f"{'='*55}")

    try:
        # -- 1. Inspect
        inspector = DataInspector(df, target_col)
        is_valid, msg = inspector.sanity_check()
        if not is_valid:
            return render(request, 'dashboard/error.html', {
                'error': msg,
                'back_url': 'dashboard',
            })

        problem_type = inspector.detect_problem_type()
        col_types    = inspector.get_column_types()
        date_col     = col_types['date_cols'][0] if col_types['date_cols'] else None
        print(f"  problem={problem_type}  date_col={date_col}")

        # -- 2. Preprocess
        engineer = FeatureEngineer(
            df, target_col, date_col=date_col, problem_type=problem_type
        )
        training_df = engineer.preprocess()

        print(f"  training_df cols : {list(training_df.columns)}")
        print(f"  date_df cols     : {list(engineer.date_df.columns)}")
        print(f"  dropped leakage  : {engineer.dropped_leakage}")
        print(f"  dropped noise    : {engineer.dropped_noise}")
        print(f"  dropped redundant: {engineer.dropped_redundant}")

        # -- 3. Train
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

        # -- 4. Extract feature list from fitted pipeline
        num_cols, cat_cols, model_feature_cols = _extract_pipeline_features(best_model)
        print(f"  model_feature_cols={model_feature_cols}")

        # -- 5. Determine user input columns
        user_input_cols = [
            c for c in model_feature_cols
            if c in original_csv_cols and c != target_col
        ]
        print(f"  user_input_cols={user_input_cols}")

        # -- 6. Visualize
        try:
            plots = DataVisualizer(training_df, target_col, problem_type).generate_all()
        except Exception as e:
            print(f"  viz failed: {e}")
            plots = {}

        # -- 7. Diagnose
        diagnosis = ModelDiagnoser(results, engineer, target_col).get_diagnosis()

        # -- 8. Save model + metadata
        model_path, meta_path = _model_paths(dataset_id)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        fingerprint = _compute_data_fingerprint(dataset.file.path, target_col)
        existing_model_path = _find_existing_model_by_fingerprint(fingerprint)

        if existing_model_path and existing_model_path != model_path:
            model_path = existing_model_path
            print(f"  Reusing existing model -> {existing_model_path}")
        else:
            joblib.dump(best_model, model_path)
            print(f"  model saved -> {model_path}")

        meta = {
            'target_col':          target_col,
            'date_col':            date_col,
            'original_csv_cols':   original_csv_cols,
            'model_feature_cols':  model_feature_cols,
            'num_cols':            num_cols,
            'cat_cols':            cat_cols,
            'user_input_cols':     user_input_cols,
            'problem_type':        problem_type,
            'file_path':           dataset.file.path,
            'model_path':          model_path,
            'has_trend':           date_col is not None and engineer.reference_date is not None,
            'trend_freq':          engineer.trend_freq,
            'reference_date':      str(engineer.reference_date) if engineer.reference_date else None,
            'data_fingerprint':    fingerprint,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # Save target_col to model
        dataset.target_col = target_col
        dataset.save()

        # -- 9. Build context
        feature_types = {
            col: ('number' if pd.api.types.is_numeric_dtype(training_df[col]) else 'text')
            for col in user_input_cols
        }
        reference_data = training_df[user_input_cols].head(5).to_dict(orient='records')

        context = {
            'results':      results,
            'best_metrics': best_metrics,
            'diagnosis':    diagnosis,
            'target':       target_col,
            'problem_type': problem_type,
            'dataset_id':   dataset_id,
            'dataset':      dataset,
            'feature_columns': user_input_cols,
            'feature_types':   feature_types,
            'reference_data':  reference_data,
            'has_trend':   meta['has_trend'],
            'date_col':    date_col or '',
            'target_col':  target_col,
            'plots':       plots,
        }
        return render(request, 'dashboard/result.html', context)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, 'dashboard/error.html', {
            'error': f"ML Pipeline Failed: {str(e)}",
            'back_url': 'dashboard',
        })


# ──────────────────────────────────────────────────────────────
# PREDICT (manual feature entry)
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def predict_view(request, dataset_id):
    """
    Standard prediction: user fills in only the columns the model was
    trained on (all from the original CSV, no date-derived cols).
    """
    dataset = _get_user_dataset(request, dataset_id)

    if request.method != 'POST':
        return redirect('select_target', dataset_id=dataset_id)

    model_path, meta_path = _model_paths(dataset_id)

    if not os.path.exists(model_path):
        return render(request, 'dashboard/error.html', {
            'error': "Model not found. Please re-train.",
            'back_url': 'dashboard',
        })
    if not os.path.exists(meta_path):
        return render(request, 'dashboard/error.html', {
            'error': "Model metadata missing. Please re-train.",
            'back_url': 'dashboard',
        })

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        model              = joblib.load(model_path)
        user_input_cols    = meta['user_input_cols']
        model_feature_cols = meta['model_feature_cols']

        original_df = pd.read_csv(dataset.file.path)

        # Parse user input
        input_data = {}
        for col in user_input_cols:
            raw = request.POST.get(col, '').strip()
            if col in original_df.columns and pd.api.types.is_numeric_dtype(original_df[col]):
                try:
                    input_data[col] = float(raw)
                except (ValueError, TypeError):
                    input_data[col] = float(original_df[col].median())
            else:
                input_data[col] = raw if raw else (
                    str(original_df[col].mode().iloc[0]) if col in original_df.columns else ''
                )

        row      = {col: input_data.get(col, np.nan) for col in model_feature_cols}
        input_df = pd.DataFrame([row])

        prediction = model.predict(input_df)[0]
        if isinstance(prediction, (int, float, np.integer, np.floating)):
            prediction = round(float(prediction), 2)

        return render(request, 'dashboard/prediction_result.html', {
            'inputs':     input_data,
            'prediction': prediction,
            'dataset_id': dataset_id,
            'dataset':    dataset,
            'target_col': meta.get('target_col', ''),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, 'dashboard/error.html', {
            'error': f"Prediction failed: {str(e)}",
            'back_url': 'dashboard',
        })


# ──────────────────────────────────────────────────────────────
# TREND FORECAST (date only — separate path)
# ──────────────────────────────────────────────────────────────

@login_required(login_url='login')
def predict_future_trend(request, dataset_id):
    """
    Trend-based prediction using dataset_id (secure — no file paths in form).
    """
    dataset = _get_user_dataset(request, dataset_id)

    if request.method != 'POST':
        return render(request, 'dashboard/error.html', {
            'error': 'Invalid request method.',
            'back_url': 'dashboard',
        })

    try:
        future_date_str = request.POST.get('future_date', '').strip()

        if not future_date_str:
            return render(request, 'dashboard/error.html', {
                'error': "Please select a future date.",
                'back_url': 'dashboard',
            })

        # Load metadata securely from dataset_id
        model_path, meta_path = _model_paths(dataset_id)

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            return render(request, 'dashboard/error.html', {
                'error': "Model not found. Please re-train first.",
                'back_url': 'dashboard',
            })

        with open(meta_path) as f:
            meta = json.load(f)

        file_path  = dataset.file.path  # Secure: from DB, not from user input
        date_col   = meta.get('date_col', '')
        target_col = meta.get('target_col', '')

        if not date_col or not target_col:
            return render(request, 'dashboard/error.html', {
                'error': "Model metadata is incomplete. Please re-train.",
                'back_url': 'dashboard',
            })

        # Load model
        main_model = joblib.load(model_path)

        # Re-run preprocessing to get both dataframes
        original_df = pd.read_csv(file_path)
        engineer    = FeatureEngineer(
            original_df, target_col=target_col, date_col=date_col
        )
        engineer.preprocess()

        if engineer.reference_date is None:
            return render(request, 'dashboard/error.html', {
                'error': f"Could not detect a date sequence in column '{date_col}'.",
                'back_url': 'dashboard',
            })

        # Get exact feature list model was trained on
        _, _, model_feature_cols = _extract_pipeline_features(main_model)

        # Build the combined trend training df
        trend_train_df = engineer.get_trend_training_df()

        # Train the trend predictor
        trend_predictor = AdvancedTrendPredictor()
        trend_predictor.train_trends(trend_train_df, model_feature_cols)

        # Convert future date to time index
        future_index, year, month = get_future_time_index(
            future_date_str, engineer.reference_date, engineer.trend_freq
        )

        # Predict all feature values for that date
        predicted_features_df = trend_predictor.predict_for_date(future_index, year, month)

        # Final prediction from main model
        final_prediction = main_model.predict(predicted_features_df)[0]
        if isinstance(final_prediction, (float, np.floating)):
            final_prediction = round(float(final_prediction), 2)

        # Hide date-derived cols from display
        features_dict = {
            k: v for k, v in predicted_features_df.iloc[0].to_dict().items()
            if k not in DATE_DERIVED_COLS
        }

        return render(request, 'dashboard/trend_result.html', {
            'future_date':        future_date_str,
            'final_prediction':   final_prediction,
            'target_col':         target_col,
            'predicted_features': features_dict,
            'dataset_id':         dataset_id,
            'dataset':            dataset,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, 'dashboard/error.html', {
            'error': f"Trend prediction failed: {str(e)}",
            'back_url': 'dashboard',
        })