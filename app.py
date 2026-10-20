"""
app.py
======
Stage 4: Streamlit Web Application for Spam Email Detection System

Features:
---------
- Professional Blue-and-White responsive UI with styled metric cards and badges.
- Dynamic data loading from outputs/evaluation_metrics.json (never hardcoded).
- Dataset overview: size, class distribution, and training/testing split details.
- Genuine test-set evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
- Interactive prediction console with empty/whitespace input validation.
- Output probabilities and confidence indicators.
- Explicit notice regarding dataset source (UCI SMS Spam Collection) and domain boundaries.
"""

import json
import os
import joblib
import pandas as pd
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Spam Email Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "models", "spam_classifier_pipeline.joblib")
METRICS_FILE = os.path.join(BASE_DIR, "outputs", "evaluation_metrics.json")
CONFUSION_MATRIX_FILE = os.path.join(BASE_DIR, "outputs", "confusion_matrix.csv")

# Custom CSS for a clean, professional blue-and-white theme
st.markdown(
    """
    <style>
    /* Main container styling */
    .main {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Header typography */
    h1, h2, h3 {
        color: #1e3a8a !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Custom metric card */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04);
        margin-bottom: 12px;
        transition: transform 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(30, 58, 138, 0.08);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #1e40af;
        margin-top: 4px;
    }
    .metric-subtitle {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Prediction Result Cards */
    .result-card-ham {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        padding: 18px 22px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .result-card-spam {
        background-color: #fef2f2;
        border-left: 6px solid #dc2626;
        padding: 18px 22px;
        border-radius: 8px;
        margin-top: 15px;
    }

    /* Informational Notice */
    .dataset-notice {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.90rem;
        color: #1e40af;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_trained_pipeline():
    """Load the joblib pipeline containing fitted vectorizer and model."""
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_evaluation_data():
    """Load genuine model evaluation metrics and confusion matrix."""
    if not os.path.exists(METRICS_FILE):
        return None, None

    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm_df = None
    if os.path.exists(CONFUSION_MATRIX_FILE):
        cm_df = pd.read_csv(CONFUSION_MATRIX_FILE, index_col=0)

    return metrics, cm_df


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
        st.title("System Controls")
        st.markdown("**Model:** Multinomial Naive Bayes")
        st.markdown("**Features:** TF-IDF (Unigrams + Bigrams)")
        st.markdown("**Classifier Split:** 80% Train / 20% Test")
        st.markdown("---")

        st.subheader("Quick Presets")
        preset_choice = st.selectbox(
            "Load a sample message to test:",
            [
                "-- Select a sample --",
                "Legitimate Meeting Notice",
                "Lottery Scam / Urgent Action",
                "Casual Friend Text",
                "Bank Account Suspension Alert",
            ],
        )

        preset_texts = {
            "Legitimate Meeting Notice": "Hi team, please find the updated slides attached for tomorrow's 10 AM project review.",
            "Lottery Scam / Urgent Action": "URGENT! You have won a 1000 GBP cash prize or a holiday voucher! Call 09061701461 now to claim.",
            "Casual Friend Text": "Hey, let me know when you reach home so we can order dinner.",
            "Bank Account Suspension Alert": "Security Alert: Your debit card is locked due to suspicious activity. Verify credentials at http://bit.ly/secure-login",
        }

        st.markdown("---")
        st.caption("Spam Email Detection System • Machine Learning Project")

    # Main Area Header
    st.title("🛡️ Spam Email Detection System")
    st.markdown(
        "An intelligent text classification system powered by **TF-IDF Vectorization** and **Multinomial Naive Bayes**, "
        "designed to identify and filter malicious or spam communications."
    )

    # Domain / Dataset Limitation Notice
    st.markdown(
        """
        <div class="dataset-notice">
            ℹ️ <strong>Dataset & Domain Scope Notice:</strong> This model was trained on the 
            <strong>UCI SMS Spam Collection</strong> dataset (5,171 cleaned messages). While linguistic spam markers 
            (urgency, suspicious claims, financial triggers) closely mirror email spam, predictions on full-length 
            emails with HTML tags, headers, and attachments are approximations and not guaranteed to be 100% correct.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load artifacts
    pipeline = load_trained_pipeline()
    metrics, cm_df = load_evaluation_data()

    if pipeline is None or metrics is None:
        st.error(
            "⚠️ Trained model artifacts not found! Please run `python train_model.py` first to train the model and save artifacts."
        )
        st.stop()

    # Section 1: Genuine Model & Dataset Metrics Display
    st.subheader("📊 Model & Dataset Performance")
    t_metrics = metrics.get("test_metrics", {})
    d_summary = metrics.get("dataset_summary", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-value">{t_metrics.get('accuracy', 0) * 100:.2f}%</div>
                <div class="metric-subtitle">{metrics.get('test_samples', 0)} test samples</div>
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
                <div class="metric-subtitle">Minimal False Positives</div>
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
                <div class="metric-subtitle">Spam caught rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">F1-Score</div>
                <div class="metric-value">{t_metrics.get('f1_score', 0) * 100:.2f}%</div>
                <div class="metric-subtitle">Harmonic balance</div>
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
                <div class="metric-subtitle">{d_summary.get('ham_percentage', 0)}% Ham / {d_summary.get('spam_percentage', 0)}% Spam</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Dataset and Confusion Matrix Expander
    with st.expander("🔍 View Detailed Dataset Distribution & Confusion Matrix", expanded=False):
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("##### Dataset Breakdown")
            dist_data = {
                "Category": ["Raw Total", "Duplicates Removed", "Cleaned Total", "Cleaned Ham (Legitimate)", "Cleaned Spam"],
                "Count": [
                    d_summary.get("raw_total", 0),
                    d_summary.get("duplicates_removed", 0),
                    d_summary.get("cleaned_total", 0),
                    d_summary.get("cleaned_distribution", {}).get("ham", 0),
                    d_summary.get("cleaned_distribution", {}).get("spam", 0),
                ],
            }
            st.dataframe(pd.DataFrame(dist_data), hide_index=True, use_container_width=True)

        with exp_col2:
            st.markdown("##### Test Set Confusion Matrix (Unseen 1,035 Records)")
            if cm_df is not None:
                st.dataframe(cm_df, use_container_width=True)
            cm_dict = metrics.get("confusion_matrix", {})
            st.caption(
                f"True Negatives: {cm_dict.get('true_negative', 0)} | "
                f"False Positives: {cm_dict.get('false_positive', 0)} | "
                f"False Negatives: {cm_dict.get('false_negative', 0)} | "
                f"True Positives: {cm_dict.get('true_positive', 0)}"
            )

    st.markdown("---")

    # Section 2: Interactive Prediction Interface
    st.subheader("✉️ Message Classification Console")
    st.write("Enter an email subject/body or SMS message below to analyze its probability of being spam.")

    # Populate preset if selected
    default_text = ""
    if preset_choice in preset_texts:
        default_text = preset_texts[preset_choice]

    user_input = st.text_area(
        "Enter message text to analyze:",
        value=default_text,
        height=140,
        placeholder="Type or paste the contents of the email or message here...",
    )

    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
        predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Clear Input", use_container_width=False):
            st.rerun()

    if predict_clicked:
        cleaned_text = user_input.strip()

        # Validation for empty messages
        if not cleaned_text:
            st.warning("⚠️ Input is empty! Please enter text before requesting a prediction.")
        else:
            # Model inference
            prediction = pipeline.predict([cleaned_text])[0]
            probabilities = pipeline.predict_proba([cleaned_text])[0]
            ham_prob = probabilities[0] * 100
            spam_prob = probabilities[1] * 100

            st.markdown("#### Analysis Result")
            res_col1, res_col2 = st.columns([3, 2])

            with res_col1:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-card-spam">
                            <h3 style="color:#b91c1c; margin:0 0 8px 0;">🚨 Predicted: SPAM</h3>
                            <p style="margin:0; color:#450a0a;">This message exhibits linguistic patterns strongly associated with unsolicited, fraudulent, or commercial spam.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-ham">
                            <h3 style="color:#15803d; margin:0 0 8px 0;">✅ Predicted: NOT SPAM (HAM)</h3>
                            <p style="margin:0; color:#052e16;">This message exhibits patterns typical of legitimate, personal, or transactional communication.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with res_col2:
                st.markdown("##### Confidence Probability")
                st.write(f"**Not Spam (Ham):** `{ham_prob:.2f}%`")
                st.progress(int(round(ham_prob)))
                st.write(f"**Spam:** `{spam_prob:.2f}%`")
                st.progress(int(round(spam_prob)))

            st.caption(
                "Note: Machine learning predictions are probabilistic inferences based on training patterns and should be treated as guidance rather than absolute truth."
            )


if __name__ == "__main__":
    main()
