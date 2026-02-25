import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier

class ModelTrainer:
    def __init__(self, df, target_col, preprocessor, col_types, problem_type='regression'):
        self.df = df
        self.target_col = target_col
        self.preprocessor = preprocessor
        self.col_types = col_types
        self.problem_type = problem_type

    def get_models_config(self):
        if self.problem_type == 'classification':
            return {
                "Logistic Regression": {
                    "model": LogisticRegression(max_iter=1000),
                    "params": {"model__C": [0.1, 1.0, 10.0]}
                },
                "Random Forest": {
                    "model": RandomForestClassifier(random_state=42),
                    "params": {"model__n_estimators": [50, 100], "model__max_depth": [5, 10, 20]}
                },
                "Gradient Boosting": {
                    "model": GradientBoostingClassifier(random_state=42),
                    "params": {"model__learning_rate": [0.01, 0.05, 0.1], "model__n_estimators": [50, 100]}
                }
            }
        else:
            return {
                "Ridge Regression": {
                    "model": Ridge(),
                    "params": {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
                },
                "Random Forest": {
                    "model": RandomForestRegressor(random_state=42),
                    "params": {"model__n_estimators": [50, 100], "model__max_depth": [5, 10, 20]}
                },
                "Gradient Boosting": {
                    "model": GradientBoostingRegressor(random_state=42),
                    "params": {"model__learning_rate": [0.01, 0.05, 0.1], "model__n_estimators": [50, 100]}
                }
            }

    def train(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # GUARD: Check if X is empty
        if X.shape[1] == 0:
            raise ValueError("All features were dropped during preprocessing! The model has no data to train on.")

        # Split Strategy
        if self.problem_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scoring_metric = 'accuracy'
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)
            scoring_metric = 'r2'

        models_config = self.get_models_config()
        results = {}
        best_score = -float('inf')
        best_model_pipeline = None
        best_model_name = ""

        for name, config in models_config.items():
            print(f"⚡ Tuning {name} ({self.problem_type})...")
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('model', config['model'])
            ])
            search = RandomizedSearchCV(
                pipeline, config['params'], n_iter=5, cv=cv_strategy, 
                scoring=scoring_metric, n_jobs=-1, random_state=42
            )
            
            search.fit(X_train, y_train)
            tuned_model = search.best_estimator_
            preds = tuned_model.predict(X_test)
            
            metrics = {}
            if self.problem_type == 'classification':
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')
                metrics = {"Accuracy": round(acc, 4), "F1 Score": round(f1, 4)}
                score_to_compare = acc
            else:
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                metrics = {"MAE": round(mae, 2), "R2": round(r2, 4)}
                score_to_compare = r2
            
            # metrics["Best Params"] = search.best_params_
            metrics["best_params"] = search.best_params_  # <--- Underscore works perfectly
            results[name] = metrics

            if score_to_compare > best_score:
                best_score = score_to_compare
                best_model_pipeline = tuned_model
                best_model_name = name

        return {
            "problem_type": self.problem_type,
            "best_model_name": best_model_name,
            "best_model": best_model_pipeline,
            "metrics": results,
            "X_test": X_test,
            "y_test": y_test
        }