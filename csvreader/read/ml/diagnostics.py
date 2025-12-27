class ModelDiagnoser:
    def __init__(self, metrics, engineer, target_col):
        self.r2 = metrics.get('R2', metrics.get('Accuracy', 0)) # Handle both metrics
        self.engineer = engineer
        self.target_col = target_col
        
    def get_diagnosis(self):
        # 1. SUCCESS CASE
        if self.r2 > 0.6:
            return {
                "status": "success",
                "title": "✅ Model Trained Successfully",
                "message": f"Great! We found strong predictive patterns for '{self.target_col}'."
            }

        # 2. FAILURE/WARNING CASE
        reasons = []
        
        # A. Leakage (Cheating)
        if self.engineer.dropped_leakage:
            reasons.append(
                f"🚫 **Removed Cheating Features:** We dropped {self.engineer.dropped_leakage}. "
                "Their Mutual Information score was too high (>0.95), meaning they predict the target too perfectly (Leakage)."
            )
            
        # B. Noise (Useless)
        if self.engineer.dropped_noise:
            reasons.append(
                f"🗑️ **Removed Noise:** We dropped {self.engineer.dropped_noise}. "
                f"Their dependency on '{self.target_col}' was nearly zero (<0.01). They provide no useful information."
            )

        # C. Default
        if not reasons:
            reasons.append("The data seems random. No strong dependencies were found even in the kept features.")

        return {
            "status": "warning",
            "title": "⚠️ Low Accuracy Explained",
            "message": (
                f"The model score is {self.r2}. Here is what happened:\n\n" + 
                "\n\n".join(reasons)
            ),
            "suggestion": "Try collecting features that have a stronger causal relationship with the target."
        }