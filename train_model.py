"""
train_model.py
==============
Email Spam Detection Model Training & Evaluation Pipeline.

Dataset:
--------
SpamAssassin Public Corpus (Easy Ham + Spam)
- Loads real email files from data/easy_ham/ and data/spam/
- Parses MIME headers, extracts subject and body, safely handles multipart/HTML
- Deduplicates messages by subject + body hash
- Stratified 80/20 train/test split (random_state=42)
- Scikit-learn Pipeline:
    - TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.95)
    - MultinomialNB(alpha=0.1)
- Fits strictly on training split (no data leakage)
- Evaluates on unseen test partition
- Generates:
    - models/spam_classifier_pipeline.joblib
    - outputs/evaluation_metrics.json
    - outputs/confusion_matrix.csv
    - outputs/confusion_matrix.png (Seaborn / Matplotlib heatmap)
"""

import json
import os
import sys
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless script execution
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

from email_parser import combine_subject_and_body, parse_raw_email

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EASY_HAM_DIR = os.path.join(DATA_DIR, "easy_ham")
SPAM_DIR = os.path.join(DATA_DIR, "spam")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_FILE = os.path.join(MODELS_DIR, "spam_classifier_pipeline.joblib")
METRICS_FILE = os.path.join(OUTPUTS_DIR, "evaluation_metrics.json")
CONFUSION_MATRIX_CSV = os.path.join(OUTPUTS_DIR, "confusion_matrix.csv")
CONFUSION_MATRIX_PNG = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")

RANDOM_STATE = 42


def load_email_corpus() -> pd.DataFrame:
    """
    Load and parse raw email messages from SpamAssassin easy_ham and spam directories.
    Never falls back to SMS data silently.
    """
    if not os.path.exists(EASY_HAM_DIR) or not os.path.exists(SPAM_DIR):
        print(
            "\n" + "=" * 75 + "\n"
            "[ERROR] SpamAssassin email dataset directories not found!\n"
            f"Expected ham directory:  {EASY_HAM_DIR}\n"
            f"Expected spam directory: {SPAM_DIR}\n\n"
            "Please run 'python download_data.py' first to download and unpack the email corpus.\n"
            "Never silently falling back to the old SMS dataset.\n"
            + "=" * 75 + "\n",
            file=sys.stderr,
        )
        sys.exit(1)

    ham_files = [
        os.path.join(EASY_HAM_DIR, f)
        for f in os.listdir(EASY_HAM_DIR)
        if os.path.isfile(os.path.join(EASY_HAM_DIR, f)) and not f.startswith("cmds")
    ]
    spam_files = [
        os.path.join(SPAM_DIR, f)
        for f in os.listdir(SPAM_DIR)
        if os.path.isfile(os.path.join(SPAM_DIR, f)) and not f.startswith("cmds")
    ]

    if not ham_files or not spam_files:
        print(
            f"[ERROR] Insufficient email files: found {len(ham_files)} ham and {len(spam_files)} spam files.\n"
            "Run 'python download_data.py' to acquire the complete SpamAssassin corpus.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[STEP 1] Parsing raw RFC 822 email files...")
    print(f" -> Ham source files:  {len(ham_files)}")
    print(f" -> Spam source files: {len(spam_files)}")

    records = []

    # Parse Ham
    for path in ham_files:
        try:
            with open(path, "rb") as f:
                content = f.read()
            subject, body = parse_raw_email(content)
            full_text = combine_subject_and_body(subject, body)
            if full_text.strip():
                records.append({
                    "filename": os.path.basename(path),
                    "label": "ham",
                    "subject": subject,
                    "body": body,
                    "full_text": full_text,
                })
        except Exception:
            continue

    # Parse Spam
    for path in spam_files:
        try:
            with open(path, "rb") as f:
                content = f.read()
            subject, body = parse_raw_email(content)
            full_text = combine_subject_and_body(subject, body)
            if full_text.strip():
                records.append({
                    "filename": os.path.basename(path),
                    "label": "spam",
                    "subject": subject,
                    "body": body,
                    "full_text": full_text,
                })
        except Exception:
            continue

    df = pd.DataFrame(records)
    print(f" -> Total parsed email records: {len(df)}")
    return df


def validate_and_clean(df: pd.DataFrame):
    """
    Validate, deduplicate, and encode dataset.
    """
    print("\n[STEP 2] Validating and deduplicating email dataset...")
    raw_total = len(df)
    raw_dist = df["label"].value_counts().to_dict()
    print(f" -> Raw distribution: {raw_dist}")

    # Remove duplicates based on combined email text
    duplicates = df.duplicated(subset=["full_text"]).sum()
    print(f" -> Duplicate email messages identified: {duplicates}")
    df_clean = df.drop_duplicates(subset=["full_text"]).reset_index(drop=True)

    cleaned_total = len(df_clean)
    cleaned_dist = df_clean["label"].value_counts().to_dict()
    print(f" -> Cleaned total: {cleaned_total}")
    print(f" -> Cleaned distribution: {cleaned_dist}")

    # Binary label mapping: ham=0, spam=1
    df_clean["target"] = df_clean["label"].map({"ham": 0, "spam": 1})

    summary = {
        "dataset_name": "SpamAssassin Public Email Corpus",
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
    Perform 80/20 stratified split, train TF-IDF + MultinomialNB pipeline,
    evaluate on unseen test emails, generate metrics, and save all artifacts.
    """
    print("\n[STEP 3] Performing 80/20 stratified train/test split (random_state=42)...")
    X = df["full_text"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f" -> Training set size: {len(X_train)} emails (Ham: {(y_train == 0).sum()}, Spam: {(y_train == 1).sum()})")
    print(f" -> Testing set size:  {len(X_test)} emails (Ham: {(y_test == 0).sum()}, Spam: {(y_test == 1).sum()})")

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
                sublinear_tf=True,
            ),
        ),
        (
            "nb",
            MultinomialNB(alpha=0.1),
        ),
    ])

    print("[STEP 5] Training pipeline strictly on training emails (X_train)...")
    pipeline.fit(X_train, y_train)
    print(" -> Pipeline training complete.")

    print("\n[STEP 6] Evaluating on UNSEEN email test set (X_test)...")
    y_pred = pipeline.predict(X_test)

    # Actual test set metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    spam_precision = float(precision_score(y_test, y_pred, pos_label=1))
    spam_recall = float(recall_score(y_test, y_pred, pos_label=1))
    spam_f1 = float(f1_score(y_test, y_pred, pos_label=1))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 60)
    print("        ACTUAL TEST SET EVALUATION METRICS (EMAIL)       ")
    print("=" * 60)
    print(f" Overall Accuracy : {accuracy * 100:.2f}% ({accuracy:.4f})")
    print(f" Spam Precision   : {spam_precision * 100:.2f}% ({spam_precision:.4f})")
    print(f" Spam Recall      : {spam_recall * 100:.2f}% ({spam_recall:.4f})")
    print(f" Spam F1-Score    : {spam_f1 * 100:.2f}% ({spam_f1:.4f})")
    print("-" * 60)
    print(" Confusion Matrix:")
    print(f"   True Negative  (Ham correctly identified) : {tn}")
    print(f"   False Positive (Ham misclassified as Spam): {fp}")
    print(f"   False Negative (Spam missed as Ham)       : {fn}")
    print(f"   True Positive  (Spam correctly caught)    : {tp}")
    print("=" * 60)
    print("\nDetailed Classification Report:\n")
    report_str = classification_report(y_test, y_pred, target_names=["Ham (0)", "Spam (1)"])
    print(report_str)

    # Save artifacts
    print("[STEP 7] Ensuring directories and saving outputs...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    joblib.dump(pipeline, MODEL_FILE)
    print(f" -> Trained model saved to: {MODEL_FILE}")

    # Confusion matrix CSV
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Ham", "Actual_Spam"],
        columns=["Predicted_Ham", "Predicted_Spam"],
    )
    cm_df.to_csv(CONFUSION_MATRIX_CSV)
    print(f" -> Confusion matrix CSV saved to: {CONFUSION_MATRIX_CSV}")

    # Confusion matrix Heatmap Image
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Ham", "Predicted Spam"],
            yticklabels=["Actual Ham", "Actual Spam"],
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
        )
        plt.title("Email Spam Detection — Test Confusion Matrix", fontsize=12, pad=12, color="#1e3a8a", weight="bold")
        plt.ylabel("Actual Label", fontsize=10, weight="bold")
        plt.xlabel("Predicted Label", fontsize=10, weight="bold")
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PNG, dpi=200)
        plt.close()
        print(f" -> Confusion matrix chart saved to: {CONFUSION_MATRIX_PNG}")
    except Exception as exc:
        print(f" -> Note: Could not generate confusion matrix PNG: {exc}")

    # Metrics JSON
    metrics_payload = {
        "model_name": "Multinomial Naive Bayes with TF-IDF Vectorizer",
        "dataset_name": "SpamAssassin Public Email Corpus",
        "dataset_source": "https://spamassassin.apache.org/old/publiccorpus/",
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
    print(f" -> Genuine evaluation metrics JSON saved to: {METRICS_FILE}")

    return pipeline, metrics_payload


def main():
    df = load_email_corpus()
    df_clean, summary = validate_and_clean(df)
    train_and_evaluate(df_clean, summary)
    print("\n[SUCCESS] Email Spam Detection model training pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
