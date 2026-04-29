import pandas as pd
import numpy as np
from datetime import datetime

#this class aggregates test-set metrics and packages them for display
class EvaluationReport:

    def __init__(self):
        self.accuracy: float = 0.0
        self.f1_score: float = 0.0
        self.confusion_matrix: list = []
        self._records: list = []

    def from_dict(self, metrics: dict) -> None:
        self.accuracy = metrics.get("accuracy", 0.0)
        self.f1_score = metrics.get("f1", 0.0)
        self.precision = metrics.get("precision", 0.0)
        self.recall = metrics.get("recall", 0.0)
        self.confusion_matrix = metrics.get("confusion_matrix", [])

    def add_record(self, email_text: str, true_label: str, predicted_label: str, confidence: float) -> None:
        self._records.append(
            {
                "timestamp": datetime.now().isoformat(),
                "email_snippet": email_text[:80],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": round(confidence, 4),
                "correct": true_label == predicted_label,
            }
        )

    def generate_report(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "confusion_matrix": self.confusion_matrix,
            "total_records": len(self._records),
        }

    def export_csv(self, path: str = "evaluation_results.csv") -> None:
        if self._records:
            pd.DataFrame(self._records).to_csv(path, index=False)
            print(f"Results exported to {path}")
        else:
            print("No records to export.")