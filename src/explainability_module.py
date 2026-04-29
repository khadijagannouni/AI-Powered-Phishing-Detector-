import numpy as np
import shap
from lime.lime_text import LimeTextExplainer

# Produces local (LIME) and global (SHAP) explanations
class ExplainabilityModule:

    def __init__(self):
        self.lime_exp = None
        self.shap_values = None
        self.explainer_lime = LimeTextExplainer(class_names=["Ham", "Phish"])

    #  LIME : local per-email explanation                                  
    #this function uses LIME to produce per-token importance scores

    def explain_lime(self, raw_text: str, predict_fn) -> dict:
       
        self.lime_exp = self.explainer_lime.explain_instance(
            raw_text,
            predict_fn,
            num_features=15,
            num_samples=500,
        )
        return dict(self.lime_exp.as_list())

    #  SHAP : global feature importance  
    #  Computes SHAP values for a sample of feature vectors.
                                  
    def explain_shap(self, model, X_sample: np.ndarray) -> np.ndarray:
       
        explainer = shap.LinearExplainer(model, X_sample) \
            if hasattr(model, "coef_") \
            else shap.TreeExplainer(model)
        self.shap_values = explainer.shap_values(X_sample)
        return self.shap_values

    #  HTML highlight                                                      
    def highlight_tokens(self, raw_text: str, token_weights: dict) -> str:
        
        html_tokens = []
        for word in raw_text.split():
            clean = word.lower().strip(".,!?;:")
            score = token_weights.get(clean, 0.0)
            if score > 0.05:
                intensity = min(int(score * 800), 200)
                color = f"rgba(255,{80 - intensity},{80 - intensity},0.6)"
                html_tokens.append(
                    f'<span style="background-color:{color};border-radius:3px;padding:1px 3px">{word}</span>'
                )
            elif score < -0.05:
                intensity = min(int(abs(score) * 800), 200)
                color = f"rgba({80 - intensity},200,{80 - intensity},0.6)"
                html_tokens.append(
                    f'<span style="background-color:{color};border-radius:3px;padding:1px 3px">{word}</span>'
                )
            else:
                html_tokens.append(word)
        return " ".join(html_tokens)