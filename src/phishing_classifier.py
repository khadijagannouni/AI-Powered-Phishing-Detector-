import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# this class predicts phish/ham label with a confidence score

class PhishingClassifier:
    

    def __init__(self, model_type: str = "logistic_regression"):
        self.label: str = ""
        self.confidence: float = 0.0
        self.model_type = model_type
        self.model = self._build_model(model_type)

    def _build_model(self, model_type: str):
        if model_type == "random_forest":
            return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        return LogisticRegression(max_iter=1000, random_state=42, C=1.0)

    #  Public API                                        
    
        #Fit the classifier on training data

    def train(self, X: np.ndarray, y) -> None:
        self.model.fit(X, y)

        #Return 'Phish' or 'Ham' for a single feature vector

    def predict(self, features: np.ndarray) -> str:
        pred = self.model.predict(features.reshape(1, -1))[0]
        self.label = "Phish" if pred == 1 else "Ham"
        self.confidence = self.get_confidence(features)
        return self.label
    

        #Return the probability of the predicted class

    def get_confidence(self, features: np.ndarray) -> float:
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        return float(np.max(proba))

        #Return a dict of evaluation metrics

    def evaluate(self, X_test: np.ndarray, y_test) -> dict:
        y_pred = self.model.predict(X_test)
        return {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "f1":        round(f1_score(y_test, y_pred, average="weighted"), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
            "recall":    round(recall_score(y_test, y_pred, average="weighted"), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_pred": y_pred.tolist(),
        }

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)