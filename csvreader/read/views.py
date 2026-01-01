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

@login_required(login_url='login')
def predict_view(request, dataset_id):
    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            pred_file = request.FILES['file']
            
            # 1. Locate the saved model
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', f'model_{dataset_id}.pkl')
            
            # 2. Save uploaded CSV temporarily
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_pred.csv')
            with open(temp_path, 'wb+') as destination:
                for chunk in pred_file.chunks():
                    destination.write(chunk)
            
            # 3. Run Prediction
            try:
                predictor = BatchPredictor(model_path)
                result_df, status = predictor.predict(temp_path)
                
                if status == "Success":
                    # 4. Return CSV Download
                    response = HttpResponse(content_type='text/csv')
                    response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
                    result_df.to_csv(path_or_buf=response, index=False)
                    return response
                else:
                    return render(request, 'dashboard/error.html', {'error': status})
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

        # 7. Render Result
        context = {
            'results': results,
            'diagnosis': diagnosis,
            'target': target_col,
            'problem_type': problem_type,
            'dataset_id': dataset_id # Needed for the prediction button
        }
        return render(request, 'dashboard/result.html', context)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR IN ML PIPELINE: {str(e)}")
        import traceback
        traceback.print_exc() # Prints the full error line numbers to terminal
        return render(request, 'dashboard/error.html', {'error': f"ML Pipeline Failed: {str(e)}"})