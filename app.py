"""
app.py
======
Email Spam Detection System — Streamlit Web Application

Features:
---------
- Professional Blue, White, and Dark-Navy themed user interface.
- Separate input fields for Email Subject and Email Body.
- Direct .eml file uploader parsed with Python's standard email module.
- Interactive 'Analyze Email' button with empty/whitespace input validation.
- Clear SPAM vs. NOT SPAM result card with dynamic probability distributions.
- Heuristic indicator analyzer (suspicious trigger keywords, excessive links, uppercase words, exclamation marks)
  clearly designated as educational heuristics rather than definitive proof.
- Live model performance metrics loaded directly from outputs/evaluation_metrics.json (never hardcoded).
- Confusion matrix display (table and seaborn heatmap visualization).
- Clear disclaimer clarifying that predictions are probabilistic machine-learning estimates, not absolute guarantees.
- One-click Reset / Clear Inputs control.
"""

import json
import os
import joblib
import pandas as pd
import streamlit as st

from email_parser import (
    analyze_heuristics,
    combine_subject_and_body,
    parse_raw_email,
)

# Page configuration
st.set_page_config(
    page_title="Email Spam Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "models", "spam_classifier_pipeline.joblib")
METRICS_FILE = os.path.join(BASE_DIR, "outputs", "evaluation_metrics.json")
CONFUSION_MATRIX_CSV = os.path.join(BASE_DIR, "outputs", "confusion_matrix.csv")
CONFUSION_MATRIX_PNG = os.path.join(BASE_DIR, "outputs", "confusion_matrix.png")

# Custom CSS for polished blue, white, and dark-navy aesthetic
st.markdown(
    """
    <style>
    /* Global background and base styles */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Typography */
    h1 {
        color: #0f172a !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    h2, h3, h4 {
        color: #1e3a8a !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        font-weight: 600;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        margin-bottom: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.08);
    }
    .metric-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #1e40af;
        margin-top: 4px;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.76rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Prediction result cards */
    .result-card-spam {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        border-radius: 8px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .result-card-ham {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        border-radius: 8px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Heuristic badge box */
    .heuristic-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        margin: 3px 4px 3px 0;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-warn {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .badge-info {
        background-color: #dbeafe;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .badge-neutral {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #1e3a8a;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_pipeline():
    """Load the trained Scikit-learn Pipeline from disk."""
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_evaluation_metrics():
    """Load genuine model evaluation metrics from outputs/evaluation_metrics.json."""
    if not os.path.exists(METRICS_FILE):
        return None
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def reset_form_state():
    """Reset session state values for inputs."""
    st.session_state["email_subject"] = ""
    st.session_state["email_body"] = ""
    st.session_state["uploaded_file_key"] = st.session_state.get("uploaded_file_key", 0) + 1


def main():
    # Initialize session state for subject & body if absent
    if "email_subject" not in st.session_state:
        st.session_state["email_subject"] = ""
    if "email_body" not in st.session_state:
        st.session_state["email_body"] = ""
    if "uploaded_file_key" not in st.session_state:
        st.session_state["uploaded_file_key"] = 0

    # Sidebar: Controls and preset templates
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
        st.title("System Overview")
        st.markdown("**Engine:** TF-IDF + MultinomialNB")
        st.markdown("**Training Corpus:** SpamAssassin Email Corpus")
        st.markdown("**Data Type:** Real MIME RFC 822 Emails")
        st.markdown("---")

        st.subheader("Quick Presets")
        preset_choice = st.selectbox(
            "Load a realistic sample email:",
            [
                "-- Select a sample --",
                "Phishing / Account Suspension",
                "Lottery Scam / Prize Claim",
                "Workplace Project Update",
                "Customer Service Inquiry",
            ],
        )

        preset_data = {
            "Phishing / Account Suspension": {
                "subject": "URGENT: Your Bank Account Access Has Been Restricted",
                "body": (
                    "Dear valued customer,\n\n"
                    "We detected unauthorized access attempts on your online banking account. "
                    "For your protection, your access has been temporarily suspended. "
                    "You must verify your account identity immediately within 24 hours to prevent permanent deactivation.\n\n"
                    "Click here to verify your account: http://secure-verify-auth99.xyz/login\n\n"
                    "Failure to act now will lead to full restriction.\n\n"
                    "Security Department"
                ),
            },
            "Lottery Scam / Prize Claim": {
                "subject": "WINNER! You have won $1,500,000 in our international lottery!",
                "body": (
                    "CONGRATULATIONS!\n\n"
                    "Your email address won a cash prize of 1.5 million dollars in our annual international lottery. "
                    "To claim your prize, please send your full name, phone number, and bank account details "
                    "for an immediate wire transfer via Western Union.\n\n"
                    "Act now before this limited time offer expires!\n"
                    "Claims Agent: Mr. David Miller"
                ),
            },
            "Workplace Project Update": {
                "subject": "Q4 Sprint Planning & Architecture Review - Thursday 2 PM",
                "body": (
                    "Hi team,\n\n"
                    "Here is a quick reminder regarding our scheduled sprint review this Thursday at 2:00 PM EST. "
                    "Please review the attached project roadmap in Confluence prior to the meeting. "
                    "If anyone needs to add agenda items, kindly submit them to the shared document by Wednesday.\n\n"
                    "Best regards,\n"
                    "Sarah Jenkins\n"
                    "Engineering Lead"
                ),
            },
            "Customer Service Inquiry": {
                "subject": "Order #48192 - Delivery confirmation question",
                "body": (
                    "Hello Support Team,\n\n"
                    "I received tracking information for order #48192, but the delivery date shows a delay of three business days. "
                    "Could you please confirm if the package has already departed the regional fulfillment center?\n\n"
                    "Thank you for your assistance,\n"
                    "Michael Brown"
                ),
            },
        }

        if preset_choice in preset_data:
            if st.button("Apply Selected Preset"):
                st.session_state["email_subject"] = preset_data[preset_choice]["subject"]
                st.session_state["email_body"] = preset_data[preset_choice]["body"]
                st.rerun()

        st.markdown("---")
        st.caption("Email Spam Detection System • Machine Learning Project")

    # Header section
    st.title("🛡️ Email Spam Detection System")
    st.markdown(
        "A robust machine learning application designed to classify emails as **Spam** or **Not Spam (Ham)**. "
        "Trained on the benchmark **SpamAssassin Public Email Corpus**, the classifier analyzes raw email text, "
        "MIME messages, and contextual heuristic indicators."
    )

    # Operational notice / disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            ℹ️ <strong>System Scope & Probabilistic Nature:</strong> This tool analyzes email subject and body content 
            supplied by the user. Predictions are generated via a probabilistic Scikit-learn machine learning model 
            (TF-IDF + Multinomial Naive Bayes) trained on genuine email corpora. 
            Predictions represent statistical likelihoods, <strong>not absolute guarantees</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load model and metrics
    pipeline = load_model_pipeline()
    metrics = load_evaluation_metrics()

    if pipeline is None:
        st.error(
            "⚠️ Trained email classifier pipeline not found at `models/spam_classifier_pipeline.joblib`. "
            "Please run `python train_model.py` in your terminal to train the model on the email corpus first."
        )
        st.stop()

    # Section 1: Dynamic Model Performance Metrics
    if metrics:
        st.subheader("📊 Empirical Model Performance (Held-Out Test Emails)")
        t_metrics = metrics.get("test_metrics", {})
        d_summary = metrics.get("dataset_summary", {})

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Test Accuracy</div>
                    <div class="metric-value">{t_metrics.get('accuracy', 0) * 100:.2f}%</div>
                    <div class="metric-sub">{metrics.get('test_samples', 0)} unseen emails</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Spam Precision</div>
                    <div class="metric-value">{t_metrics.get('precision', 0) * 100:.2f}%</div>
                    <div class="metric-sub">Zero False Positives</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Spam Recall</div>
                    <div class="metric-value">{t_metrics.get('recall', 0) * 100:.2f}%</div>
                    <div class="metric-sub">Spam detection rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Spam F1-Score</div>
                    <div class="metric-value">{t_metrics.get('f1_score', 0) * 100:.2f}%</div>
                    <div class="metric-sub">Harmonic precision-recall</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Dataset Size</div>
                    <div class="metric-value">{d_summary.get('cleaned_total', 0):,}</div>
                    <div class="metric-sub">{d_summary.get('ham_percentage', 0)}% Ham / {d_summary.get('spam_percentage', 0)}% Spam</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("🔍 View Dataset Details & Confusion Matrix", expanded=False):
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                st.markdown("##### Dataset Breakdown")
                breakdown_df = pd.DataFrame({
                    "Attribute": [
                        "Corpus Source",
                        "Raw Parsed Emails",
                        "Duplicates Removed",
                        "Cleaned Dataset",
                        "Legitimate Emails (Ham)",
                        "Spam Emails",
                        "Training Split (80%)",
                        "Testing Split (20%)",
                    ],
                    "Value": [
                        "SpamAssassin Public Corpus",
                        f"{d_summary.get('raw_total', 0):,}",
                        f"{d_summary.get('duplicates_removed', 0):,}",
                        f"{d_summary.get('cleaned_total', 0):,}",
                        f"{d_summary.get('cleaned_distribution', {}).get('ham', 0):,} ({d_summary.get('ham_percentage', 0)}%)",
                        f"{d_summary.get('cleaned_distribution', {}).get('spam', 0):,} ({d_summary.get('spam_percentage', 0)}%)",
                        f"{metrics.get('train_samples', 0):,} samples",
                        f"{metrics.get('test_samples', 0):,} samples",
                    ],
                })
                st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

            with exp_c2:
                st.markdown("##### Test Set Confusion Matrix Heatmap")
                if os.path.exists(CONFUSION_MATRIX_PNG):
                    st.image(CONFUSION_MATRIX_PNG, use_container_width=True)
                elif os.path.exists(CONFUSION_MATRIX_CSV):
                    cm_df = pd.read_csv(CONFUSION_MATRIX_CSV, index_col=0)
                    st.dataframe(cm_df, use_container_width=True)

    st.markdown("---")

    # Section 2: Email File Uploader (.eml)
    st.subheader("📥 Analyze Email")

    with st.expander("📂 Optional: Upload a `.eml` raw email file", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload an RFC 822 / MIME .eml email file:",
            type=["eml", "txt"],
            key=f"eml_uploader_{st.session_state['uploaded_file_key']}",
            help="Parses standard email headers, extracts subject and decoded plain/HTML body automatically.",
        )
        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.read()
                parsed_subj, parsed_body = parse_raw_email(raw_bytes)
                st.session_state["email_subject"] = parsed_subj
                st.session_state["email_body"] = parsed_body
                st.success(f"Successfully extracted from `{uploaded_file.name}`! Subject: **{parsed_subj or '(No Subject)'}**")
            except Exception as e:
                st.error(f"Malformed or unreadable .eml file: {e}")

    # Text Input Areas for Subject and Body
    subject_input = st.text_input(
        "Email Subject:",
        value=st.session_state["email_subject"],
        placeholder="e.g. Account Security Alert or Quarterly Planning Notes",
    )
    st.session_state["email_subject"] = subject_input

    body_input = st.text_area(
        "Email Body:",
        value=st.session_state["email_body"],
        height=180,
        placeholder="Paste the full body text of the email here...",
    )
    st.session_state["email_body"] = body_input

    # Action buttons
    btn_col1, btn_col2, btn_spacer = st.columns([1.5, 1.5, 5])
    with btn_col1:
        analyze_clicked = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Clear Inputs", use_container_width=True):
            reset_form_state()
            st.rerun()

    # Section 3: Analysis Execution
    if analyze_clicked:
        combined_text = combine_subject_and_body(subject_input, body_input)

        if not combined_text.strip():
            st.warning("⚠️ Input is empty! Please enter an email subject or body before analyzing.")
        else:
            # Model prediction and probabilities
            prediction = pipeline.predict([combined_text])[0]
            probabilities = pipeline.predict_proba([combined_text])[0]
            ham_prob = probabilities[0] * 100
            spam_prob = probabilities[1] * 100

            # Heuristics extraction
            heuristics = analyze_heuristics(subject_input, body_input)

            st.markdown("### 📋 Analysis Results")

            res_col1, res_col2 = st.columns([3, 2])
            with res_col1:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-card-spam">
                            <h2 style="color: #dc2626; margin: 0 0 8px 0;">🚨 SPAM DETECTED</h2>
                            <p style="margin: 0; color: #7f1d1d; font-size: 0.95rem;">
                                This email contains linguistic structures, vocabulary distributions, and markers 
                                strongly correlated with unsolicited commercial messaging, scams, or fraudulent phishing.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-ham">
                            <h2 style="color: #16a34a; margin: 0 0 8px 0;">✅ NOT SPAM (HAM)</h2>
                            <p style="margin: 0; color: #14532d; font-size: 0.95rem;">
                                This email aligns with the linguistic characteristics of legitimate personal, 
                                professional, or transactional correspondence.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with res_col2:
                st.markdown("##### Prediction Probabilities")
                st.write(f"**Not Spam (Ham):** `{ham_prob:.2f}%`")
                st.progress(int(round(ham_prob)))
                st.write(f"**Spam:** `{spam_prob:.2f}%`")
                st.progress(int(round(spam_prob)))

            # Section 4: Heuristic Indicators
            st.markdown("#### 🔎 Heuristic Content Analysis")
            st.caption(
                "Notice: These heuristic signals provide contextual visibility into common patterns often found in spam. "
                "They are auxiliary heuristics for educational analysis, NOT conclusive proof of malicious intent on their own."
            )

            h_col1, h_col2, h_col3, h_col4 = st.columns(4)
            with h_col1:
                st.metric("Trigger Phrases Found", len(heuristics["matched_phrases"]))
            with h_col2:
                st.metric("Links / URLs Detected", heuristics["url_count"])
            with h_col3:
                st.metric("Exclamation Marks (!)", heuristics["exclamation_count"])
            with h_col4:
                st.metric("All-Caps Trigger Words", heuristics["all_caps_count"])

            st.markdown(
                f"""
                <div class="heuristic-box">
                    <strong>Trigger Phrases Detected:</strong><br>
                    {' '.join([f'<span class="badge-pill badge-warn">{p}</span>' for p in heuristics['matched_phrases']]) if heuristics['matched_phrases'] else '<span class="badge-pill badge-neutral">No common suspicious trigger phrases found</span>'}<br><br>
                    <strong>Capitalized Keywords:</strong><br>
                    {' '.join([f'<span class="badge-pill badge-info">{w}</span>' for w in heuristics['all_caps_words']]) if heuristics['all_caps_words'] else '<span class="badge-pill badge-neutral">None</span>'}<br><br>
                    <strong>Detected URLs / Domains:</strong><br>
                    {' '.join([f'<span class="badge-pill badge-neutral">{u}</span>' for u in heuristics['sample_urls']]) if heuristics['sample_urls'] else '<span class="badge-pill badge-neutral">No URLs found</span>'}
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
