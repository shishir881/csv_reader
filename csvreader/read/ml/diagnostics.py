import pandas as pd
import numpy as np

class ModelDiagnoser:
    def __init__(self, results, engineer, target_col):
        """
        results: The full results dict returned by trainer.train()
        """
        self.best_model_name = results['best_model_name']
        self.metrics = results['metrics'][self.best_model_name]
        self.best_pipeline = results['best_model'] # The actual trained pipeline
        self.engineer = engineer
        self.target_col = target_col
        self.r2 = self.metrics.get('R2', self.metrics.get('Accuracy', 0))

    def get_feature_importance(self):
        """Extracts which features the model actually used."""
        try:
            # 1. Get the actual model from the pipeline
            model_step = self.best_pipeline.named_steps['model']
            
            # 2. Get feature names from preprocessor
            preprocessor = self.best_pipeline.named_steps['preprocessor']
            
            # This is tricky in Scikit-Learn pipelines. We try to reconstruct names.
            # (Simplified approach for robustness)
            if hasattr(model_step, 'feature_importances_'):
                importances = model_step.feature_importances_
                
                # We simply return the Top 3 most important indices if names fail
                # In a real app, getting exact column names from a Pipeline is complex
                indices = np.argsort(importances)[::-1]
                top_3_score = sum(importances[indices[:3]])
                return f"The model relied {top_3_score*100:.1f}% on the top 3 features."
            
            return "Feature importance not available for this model type."
        except:
            return "Could not extract detailed feature importance."

    def get_diagnosis(self):
        # 1. SUCCESS CASE
        if self.r2 > 0.6:
            return {
                "status": "success",
                "title": "✅ Model Trained Successfully",
                "message": f"Great! We found strong predictive patterns for '{self.target_col}'."
            }

        # 2. WARNING CASE
        reasons = []
        
        # Check Leakage
        if self.engineer.dropped_leakage:
            reasons.append(f"🚫 **Dropped Leakage:** {self.engineer.dropped_leakage} (>95% correlation).")
            
        # Check Noise (Only if user enabled filter)
        if self.engineer.dropped_noise:
            reasons.append(f"🗑️ **Dropped Noise:** {self.engineer.dropped_noise} (<1% correlation).")

        # Check Model Performance on KEPT features
        if self.r2 < 0.1:
            importance_msg = self.get_feature_importance()
            reasons.append(
                f"📉 **Weak Signal:** Even with the remaining features, the model score is low ({self.r2}).\n"
                f"This implies the data might be synthetic or random. {importance_msg}"
            )

        return {
            "status": "warning",
            "title": "⚠️ Low Accuracy Explained",
            "message": "\n\n".join(reasons) if reasons else "Model performance is low. The data appears random.",
            "suggestion": "Try collecting real-world data or checking for missing key predictors."
        }