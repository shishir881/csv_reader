import pandas as pd
import numpy as np
import os
import sys

# Setup Path
data_path = os.path.join(os.path.dirname(__file__), './dataset/Global_Cybersecurity_Threats_2015-2024.csv')
df = pd.read_csv(data_path)
target = 'Financial Loss (in Million $)'

try:
    from read.ml.utils import DataInspector
    from read.ml.preprocessing import FeatureEngineer
    from read.ml.trainer import ModelTrainer
    from read.ml.diagnostics import ModelDiagnoser
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def run_test(df, target_col):
    # print(f"\n{'='*40}\n🧪 TEST: {name}\n{'='*40}")
    
    # 1. Inspection
    print("🔍 Inspecting...")
    inspector = DataInspector(df, target_col)
# EXPLICIT SANITY CHECK (Crucial step!)
    is_valid, msg = inspector.sanity_check()
    if not is_valid:
        print(f"❌ STOPPING: {msg}")
        return

    problem_type = inspector.detect_problem_type()
    col_types = inspector.get_column_types()
    print(f"   -> Type: {problem_type}")

    # 2. Preprocessing
    print("🧹 Cleaning...")
    engineer = FeatureEngineer(df, target_col, problem_type=problem_type)
    clean_df = engineer.preprocess()
    
    if engineer.dropped_leakage: print(f"   ⚠️ LEAKAGE DROPPED: {engineer.dropped_leakage}")
    if engineer.dropped_noise:   print(f"   🗑️ NOISE DROPPED: {engineer.dropped_noise}")

    # 3. Training
    print("🏋️ Training...")
    try:
        pipeline = engineer.get_sklearn_pipeline(col_types['num_cols'], col_types['cat_cols'])
        trainer = ModelTrainer(clean_df, target_col, pipeline, col_types, problem_type=problem_type)
        results = trainer.train()
        
        best_model = results['best_model_name']
        metrics = results['metrics'][best_model]
        print(f"   -> Best: {best_model} | Score: {metrics}")

        # 4. Diagnostics
        print("🩺 Diagnosing...")
        diagnoser = ModelDiagnoser(metrics, engineer, target_col)
        report = diagnoser.get_diagnosis()
        print(f"   -> Report: {report['title']}")
        print(f"   -> Message: {report['message']}")

    except Exception as e:
        print(f"❌ TRAINING FAILED: {str(e)}")

if __name__ == "__main__":
    run_test(df, target)