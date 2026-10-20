"""
train_model.py
==============
Spam Email / SMS Detection Model Training Script

Pipeline & Workflow:
--------------------
1. Load 'data/SMSSpamCollection' as a tab-separated dataset (columns: 'label', 'message').
2. Validate and clean dataset: remove missing values and eliminate duplicate records.
3. Encode labels (ham -> 0, spam -> 1).
4. Perform an 80/20 stratified train-test split with random_state=42.
5. Build a Scikit-learn Pipeline with:
   - TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
   - MultinomialNB(alpha=0.1)
6. Fit the pipeline strictly on training data (no data leakage).
7. Evaluate test set: accuracy, spam precision, spam recall, F1 score, confusion matrix.
8. Persist the trained pipeline to 'models/spam_classifier_pipeline.joblib'.
9. Save calculated metrics to 'outputs/evaluation_metrics.json' and confusion matrix to 'outputs/confusion_matrix.csv'.
10. Ensure directories are created automatically and errors are communicated clearly.
"""

import json
import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "SMSSpamCollection")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_FILE = os.path.join(MODELS_DIR, "spam_classifier_pipeline.joblib")
METRICS_FILE = os.path.join(OUTPUTS_DIR, "evaluation_metrics.json")
CONFUSION_MATRIX_FILE = os.path.join(OUTPUTS_DIR, "confusion_matrix.csv")

RANDOM_STATE = 42


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load tab-separated SMS Spam Collection dataset with 'label' and 'message' columns.
    Raises descriptive error if file is missing or invalid.
    """
    if not os.path.exists(filepath):
        print(
            f"\n[ERROR] Dataset not found at: '{filepath}'.\n"
            "Please ensure 'data/SMSSpamCollection' exists or run 'python download_data.py' first.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[STEP 1] Loading dataset from: {filepath}")
    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=["label", "message"],
            quoting=3,  # QUOTE_NONE prevents issues with quotes in messages
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"\n[ERROR] Failed to read dataset file: {exc}\n", file=sys.stderr)
        sys.exit(1)

    if df.empty or "label" not in df.columns or "message" not in df.columns:
        print(
            "\n[ERROR] Dataset is empty or does not contain required 'label' and 'message' columns.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f" -> Raw rows loaded: {len(df)}")
    return df


def validate_and_clean(df: pd.DataFrame):
    """
    Validate and clean dataset:
    - Reports raw distribution
    - Drops missing/null records
    - Drops duplicate messages
    - Maps labels: 'ham' -> 0, 'spam' -> 1
    """
    print("\n[STEP 2] Validating and cleaning data...")
    raw_total = len(df)
    raw_dist = df["label"].value_counts().to_dict()
    print(f" -> Raw distribution: {raw_dist}")

    # Check and drop missing values
    missing_count = df.isnull().sum().to_dict()
    print(f" -> Missing values per column: {missing_count}")
    df_clean = df.dropna(subset=["label", "message"]).copy()

    # Deduplicate messages
    duplicates = df_clean.duplicated(subset=["message"]).sum()
    print(f" -> Duplicate messages found: {duplicates}")
    df_clean = df_clean.drop_duplicates(subset=["message"]).reset_index(drop=True)

    cleaned_total = len(df_clean)
    cleaned_dist = df_clean["label"].value_counts().to_dict()
    print(f" -> Cleaned records: {cleaned_total}")
    print(f" -> Cleaned distribution: {cleaned_dist}")

    # Validate classes
    valid_labels = {"ham", "spam"}
    present_labels = set(df_clean["label"].unique())
    if not present_labels.issubset(valid_labels):
        print(
            f"\n[ERROR] Unexpected class labels found: {present_labels - valid_labels}. "
            "Expected only 'ham' and 'spam'.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Encode target: ham -> 0, spam -> 1
    df_clean["target"] = df_clean["label"].map({"ham": 0, "spam": 1})

    summary = {
        "raw_total": int(raw_total),
        "raw_distribution": {k: int(v) for k, v in raw_dist.items()},
        "duplicates_removed": int(duplicates),
        "cleaned_total": int(cleaned_total),
        "cleaned_distribution": {k: int(v) for k, v in cleaned_dist.items()},
        "ham_percentage": round((cleaned_dist.get("ham", 0) / cleaned_total) * 100, 2),
        "spam_percentage": round((cleaned_dist.get("spam", 0) / cleaned_total) * 100, 2),
    }

    return df_clean, summary


def train_and_evaluate(df: pd.DataFrame, dataset_summary: dict):
    """
    Split data (80/20 stratified), fit TF-IDF + MultinomialNB pipeline,
    evaluate on unseen test data, and persist model and metrics.
    """
    print("\n[STEP 3] Performing 80/20 stratified train-test split (random_state=42)...")
    X = df["message"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f" -> Training samples: {len(X_train)} (Ham: {(y_train == 0).sum()}, Spam: {(y_train == 1).sum()})")
    print(f" -> Testing samples:  {len(X_test)} (Ham: {(y_test == 0).sum()}, Spam: {(y_test == 1).sum()})")

    print("\n[STEP 4] Constructing Pipeline (TfidfVectorizer + MultinomialNB)...")
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                lowercase=True,
                max_df=0.95,
                min_df=2,
            ),
        ),
        (
            "nb",
            MultinomialNB(alpha=0.1),
        ),
    ])

    print("[STEP 5] Training pipeline on X_train...")
    pipeline.fit(X_train, y_train)
    print(" -> Pipeline training complete.")

    print("\n[STEP 6] Evaluating on UNSEEN test set (X_test)...")
    y_pred = pipeline.predict(X_test)

    # Calculate actual test metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    spam_precision = float(precision_score(y_test, y_pred, pos_label=1))
    spam_recall = float(recall_score(y_test, y_pred, pos_label=1))
    spam_f1 = float(f1_score(y_test, y_pred, pos_label=1))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Print calculated metrics
    print("\n" + "=" * 55)
    print("           ACTUAL CALCULATED TEST SET METRICS           ")
    print("=" * 55)
    print(f" Accuracy       : {accuracy * 100:.2f}% ({accuracy:.4f})")
    print(f" Spam Precision : {spam_precision * 100:.2f}% ({spam_precision:.4f})")
    print(f" Spam Recall    : {spam_recall * 100:.2f}% ({spam_recall:.4f})")
    print(f" Spam F1-Score  : {spam_f1 * 100:.2f}% ({spam_f1:.4f})")
    print("-" * 55)
    print(" Confusion Matrix:")
    print(f"   True Negative  (Actual Ham -> Pred Ham)   : {tn}")
    print(f"   False Positive (Actual Ham -> Pred Spam)  : {fp}")
    print(f"   False Negative (Actual Spam -> Pred Ham)  : {fn}")
    print(f"   True Positive  (Actual Spam -> Pred Spam) : {tp}")
    print("=" * 55)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Ham (0)", "Spam (1)"]))

    # Step 7: Create directories if missing and save outputs
    print("[STEP 7] Ensuring directories and saving artifacts...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Save trained pipeline
    joblib.dump(pipeline, MODEL_FILE)
    print(f" -> Model pipeline saved to: {MODEL_FILE}")

    # Save confusion matrix CSV
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Ham", "Actual_Spam"],
        columns=["Predicted_Ham", "Predicted_Spam"],
    )
    cm_df.to_csv(CONFUSION_MATRIX_FILE)
    print(f" -> Confusion matrix saved to: {CONFUSION_MATRIX_FILE}")

    # Save evaluation metrics JSON
    metrics_payload = {
        "model_name": "Multinomial Naive Bayes with TF-IDF Vectorizer",
        "random_state": RANDOM_STATE,
        "dataset_summary": dataset_summary,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(spam_precision, 4),
            "recall": round(spam_recall, 4),
            "f1_score": round(spam_f1, 4),
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f" -> Evaluation metrics JSON saved to: {METRICS_FILE}")

    return pipeline, metrics_payload


def main():
    df = load_dataset(DATA_PATH)
    df_clean, summary = validate_and_clean(df)
    train_and_evaluate(df_clean, summary)
    print("\n[SUCCESS] Model training and evaluation completed successfully.\n")


if __name__ == "__main__":
    main()
