import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .forms import CustomSignupForm, CustomLoginForm, DatasetForm
from .models import Dataset

# --- ML MODULES IMPORTS (Tapaiko ML Folder bata) ---
from read.ml.utils import DataInspector
from read.ml.preprocessing import FeatureEngineer
from read.ml.trainer import ModelTrainer
from read.ml.diagnostics import ModelDiagnoser
from django.conf import settings
from django.http import HttpResponse
import os
from read.ml.predictor import BatchPredictor
from .forms import CustomSignupForm, CustomLoginForm, DatasetForm, PredictionForm
from read.ml.visualizer import DataVisualizer

# ===========================
# 🔐 AUTHENTICATION VIEWS
# ===========================

def signup_view(request):
    if request.method == 'POST':
        form = CustomSignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('upload') # Signup paxi sidhai upload ma
    else:
        form = CustomSignupForm()
    return render(request, 'registration/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = CustomLoginForm(data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            if 'next' in request.POST:
                return redirect(request.POST.get('next'))
            return redirect('upload')
    else:
        form = CustomLoginForm()
    return render(request, 'registration/login.html', {'form': form})

def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return redirect('login')

# ===========================
# 📂 APP LOGIC (Upload -> Select -> Train)
# ===========================

@login_required(login_url='login')
def upload_view(request):
    """Step 1: User uploads CSV"""
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save()
            # Upload sakepaxi Target Select garna pathaune
            return redirect('select_target', dataset_id=dataset.id)
    else:
        form = DatasetForm()
    return render(request, 'dashboard/upload.html', {'form': form})

@login_required(login_url='login')
def select_target_view(request, dataset_id):
    """Step 2: User selects which column to predict"""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    try:
        df = pd.read_csv(dataset.file.path)
        columns = df.columns.tolist()
    except Exception as e:
        return render(request, 'dashboard/error.html', {'error': f"CSV Error: {e}"})

    if request.method == 'POST':
        target = request.POST.get('target')
        # Target payepaxi balla ML Train garne function call hunxa
        return train_model_view(request, dataset_id, target)

    return render(request, 'dashboard/select_target.html', {'columns': columns, 'dataset': dataset})

# read/views.py

@login_required(login_url='login')
def predict_view(request, dataset_id):
    if request.method == 'POST':
        # 1. Load the Model
        model_path = os.path.join(settings.MEDIA_ROOT, 'models', f'model_{dataset_id}.pkl')
        if not os.path.exists(model_path):
            return render(request, 'dashboard/error.html', {'error': "Model not found."})
        
        try:
            # 2. Collect Data from Form
            # request.POST gives us a dictionary of inputs
            input_data = {}
            
            # We need to know the original columns to reconstruct the DataFrame correctly
            # (We load the dataset just to get column names/types - slight overhead but safe)
            dataset = Dataset.objects.get(id=dataset_id)
            original_df = pd.read_csv(dataset.file.path)
            
            # We assume the last trained target is NOT in the form
            # In a real app, we might store 'features' in the DB to avoid reading CSV again
            # For now, we just exclude the column that is NOT in POST data or handle all
            
            for key, value in request.POST.items():
                if key == 'csrfmiddlewaretoken': continue # Skip security token
                
                # Check original type to convert string input to float/int
                if key in original_df.columns:
                    if pd.api.types.is_numeric_dtype(original_df[key]):
                        try:
                            input_data[key] = float(value)
                        except:
                            input_data[key] = 0 # Default to 0 if empty/error
                    else:
                        input_data[key] = value

            # 3. Create a Single-Row DataFrame
            input_df = pd.DataFrame([input_data])
            
            # 4. Predict
            # We use the same FeatureEngineer but we pass target_col=None
            # Note: We need to handle preprocessing on this single row
            
            # LOAD PREDICTOR
            predictor = BatchPredictor(model_path)
            
            # The predictor expects a CSV path usually, but let's modify it or use the internal logic
            # Let's direct-call the model for simplicity here, assuming the pipeline handles it
            prediction = predictor.model.predict(input_df)[0]
            
            # Round if it's a number
            if isinstance(prediction, (int, float)):
                prediction = round(prediction, 2)

            return render(request, 'dashboard/prediction_result.html', {
                'inputs': input_data,
                'prediction': prediction,
                'dataset_id': dataset_id
            })

        except Exception as e:
             return render(request, 'dashboard/error.html', {'error': f"Prediction Error: {str(e)}"})

    return redirect('upload')

def train_model_view(request, dataset_id, target_col):
    """Step 3: ML Engine runs here (With Verbose Logging)"""
    
    # 1. Setup Data
    dataset = get_object_or_404(Dataset, id=dataset_id)
    df = pd.read_csv(dataset.file.path)

    # --- SERVER LOG START ---
    print(f"\n{'='*50}")
    print(f"🚀 SERVER LOG: ML Pipeline Started")
    print(f"📂 Dataset: {dataset.name}")
    print(f"🎯 Target: {target_col}")
    print(f"{'='*50}")

    try:
        # 2. Inspection
        print("\n🔍 Step 1: Inspecting Data...")
        inspector = DataInspector(df, target_col)
        
        # Sanity Check
        is_valid, msg = inspector.sanity_check()
        if not is_valid:
            print(f"❌ SANITY CHECK FAILED: {msg}")
            return render(request, 'dashboard/error.html', {'error': msg})

        problem_type = inspector.detect_problem_type()
        col_types = inspector.get_column_types()
        print(f"   -> Problem Type Detected: {problem_type.upper()}")
        print(f"   -> Numerical Cols: {len(col_types['num_cols'])}")
        print(f"   -> Categorical Cols: {len(col_types['cat_cols'])}")

        # 3. Preprocessing
        print("\n🧹 Step 2: Preprocessing & Feature Engineering...")
        engineer = FeatureEngineer(df, target_col, problem_type=problem_type)
        clean_df = engineer.preprocess()
        
        # Log what was dropped
        if engineer.dropped_leakage:
            print(f"   ⚠️ LEAKAGE DETECTED & DROPPED: {engineer.dropped_leakage}")
        if engineer.dropped_noise:
            print(f"   🗑️ NOISE DROPPED: {engineer.dropped_noise}")
        else:
            print("   ✅ No features dropped as Noise (Threshold=0.0)")
            
        print(f"   -> Final Shape for Training: {clean_df.shape}")

        # 4. Training
        print("\n🏋️ Step 3: Training Models...")
        pipeline = engineer.get_sklearn_pipeline(col_types['num_cols'], col_types['cat_cols'])
        trainer = ModelTrainer(clean_df, target_col, pipeline, col_types, problem_type=problem_type)
        results = trainer.train()
        
        best_model = results['best_model_name']
        metrics = results['metrics'][best_model]
        print(f"   -> ✅ Training Complete.")
        print(f"   -> 🏆 WINNER: {best_model}")
        print(f"   -> 📊 SCORE: {metrics}")


        print("📊 Generating Visualizations...")
        try:
            # We use 'clean_df' because it's clean (no NaNs) but readable
            visualizer = DataVisualizer(clean_df, target_col, problem_type)
            plots = visualizer.generate_all()
        except Exception as e:
            print(f"⚠️ Visualization Failed: {e}")
            plots = {} # Empty if fails

        

        # 5. Diagnostics
        print("\n🩺 Step 4: Running Diagnostics...")
        diagnoser = ModelDiagnoser(
            results, # Pass full results dict
            engineer, 
            target_col
        )
        diagnosis = diagnoser.get_diagnosis()
        print(f"   -> Diagnosis Status: {diagnosis['status']}")
        print(f"   -> Title: {diagnosis['title']}")

        # 6. Save Model
        import joblib, os
        from django.conf import settings
        
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'model_{dataset_id}.pkl')
        joblib.dump(results['best_model'], model_path)
        print(f"\n💾 Model Saved to: {model_path}")
        
        print(f"{'='*50}\n") # End Log

        # 1. Get Feature Columns (Only columns that survived cleaning)
        # OLD: feature_columns = [col for col in df.columns if col != target_col]
        feature_columns = [col for col in clean_df.columns if col != target_col]
        
        # 2. Get Column Data Types
        feature_types = {}
        for col in feature_columns:
            # We check types from clean_df
            if pd.api.types.is_numeric_dtype(clean_df[col]):
                feature_types[col] = 'number'
            else:
                feature_types[col] = 'text'

        # 3. Create Reference Table (Top 5 rows from clean data)
        reference_data = clean_df[feature_columns].head(5).to_dict(orient='records')

        # 7. Render Result
        context = {
            'results': results,
            'diagnosis': diagnosis,
            'target': target_col,
            'problem_type': problem_type,
            'dataset_id': dataset_id, # Needed for the prediction button

            # Pass these new things to HTML
            'feature_columns': feature_columns, 
            'feature_types': feature_types,
            'reference_data': reference_data,
            'plots': plots, # Visualizations
        }
        return render(request, 'dashboard/result.html', context)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR IN ML PIPELINE: {str(e)}")
        import traceback
        traceback.print_exc() # Prints the full error line numbers to terminal
        return render(request, 'dashboard/error.html', {'error': f"ML Pipeline Failed: {str(e)}"})