
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from src.email_processor import EmailProcessor
from src.feature_extractor import FeatureExtractor
from src.phishing_classifier import PhishingClassifier
from src.evaluation_report import EvaluationReport


def load_dataset(path: str):
    df = pd.read_csv(path)

    col_map = {}
    for col in df.columns:
        if col.lower() in ("email_text", "body", "message", "text", "content"):
            col_map[col] = "text"
        if col.lower() in ("label", "class", "target", "spam"):
            col_map[col] = "label"
    df.rename(columns=col_map, inplace=True)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns (see docs for aliases).")

    df.dropna(subset=["text", "label"], inplace=True)
    return df


def main(data_path: str, model_type: str, test_size: float = 0.2):
    print("=== AI Phishing Detector – Training Pipeline ===\n")

    # 1. Load data
    print(f"[1/5] Loading dataset from {data_path} …")
    df = load_dataset(data_path)
    print(f"      {len(df)} samples | label distribution:\n{df['label'].value_counts().to_string()}\n")

    # 2. Preprocessing
    print("[2/5] Preprocessing emails …")
    processor = EmailProcessor()
    df["cleaned"] = df["text"].apply(processor.preprocess)

    # 3. Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=42,
        stratify=df["label"].tolist(),
    )
    # Keep a small feedback split from training data
    X_train_raw, X_feedback_raw, y_train, y_feedback = train_test_split(
        X_train_raw, y_train, test_size=0.05, random_state=42
    )
    print(f"      Train: {len(X_train_raw)} | Test: {len(X_test_raw)} | Feedback: {len(X_feedback_raw)}\n")

    # 4. Feature extraction
    print("[3/5] Extracting features (TF-IDF + URL + urgency) …")
    extractor = FeatureExtractor(max_features=5000)
    X_train = extractor.fit_transform(X_train_raw)
    X_test = extractor.transform.__func__   
    # Transform individually to reuse fitted vectorizer
    X_test_arr  = extractor.transform(X_test_raw[0])  # warm-up
    X_test_arr  = [extractor.transform(t) for t in X_test_raw]
    import numpy as np
    X_test_arr  = np.vstack(X_test_arr)
    print(f"      Feature shape: {X_train.shape}\n")

    # 5. Train & evaluate
    print(f"[4/5] Training {model_type} classifier …")
    classifier = PhishingClassifier(model_type=model_type)
    classifier.train(X_train, y_train)

    print("[5/5] Evaluating on test set …")
    metrics = classifier.evaluate(X_test_arr, y_test)
    print(f"\n  Accuracy  : {metrics['accuracy']}")
    print(f"  F1 Score  : {metrics['f1']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  Confusion :\n{metrics['confusion_matrix']}\n")

    # 6. Save artifacts
    classifier.save("models/classifier.pkl")
    joblib.dump(extractor.vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(extractor, "models/feature_extractor.pkl")
    print("Artifacts saved to models/\n")
    print("=== Training complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="data/phishing_dataset.csv")
    parser.add_argument("--model", default="logistic_regression", choices=["logistic_regression", "random_forest"])
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args.data, args.model, args.test_size)