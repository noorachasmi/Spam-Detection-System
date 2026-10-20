# PROJECT REPORT
# Email Spam Detection System

**Project Type:** Applied Machine Learning Internship Project  
**Domain:** Natural Language Processing & Cyber Threat Filtering  
**Workspace:** `Spam Detection`  
**Date:** October 2026  

---

## 1. Project Title
**Email Spam Detection System Using SpamAssassin Public Corpus, TF-IDF Vectorization, and Multinomial Naive Bayes**

---

## 2. Problem Statement
Electronic mail remains the cornerstone of modern corporate, academic, and personal correspondence. However, its low transmission cost makes it a prime vehicle for malicious actors deploying phishing scams, financial advance-fee fraud, malware attachments, and deceptive marketing.

Filtering spam solely via static keyword blacklists is fragile because attackers deliberately alter vocabulary, insert random whitespace, use HTML encoding, and disguise message structures. Machine learning models, particularly probabilistic text classifiers paired with n-gram vectorization, offer an adaptive solution. However, such systems must be trained on genuine email corpora—incorporating real-world MIME structures, complex headers, and HTML payloads—while guaranteeing an exceptionally low False Positive rate to avoid discarding legitimate communications.

---

## 3. Project Objectives
1. **Transition to Genuine Email Data:** Replace short SMS text datasets with the official **Apache SpamAssassin Public Email Corpus** (~3,000 real-world emails).
2. **Robust Email & MIME Parsing:** Develop a robust parser using Python's standard `email` package and `BeautifulSoup4` to decode RFC 2047 subject headers, traverse multi-part MIME bodies, and strip HTML formatting safely.
3. **Data Deduplication & Leak-Free Splitting:** Identify and eliminate duplicate messages, performing an 80/20 stratified split (`random_state=42`) to maintain class ratios.
4. **Pipeline Construction:** Build a Scikit-learn `Pipeline` combining unigram/bigram `TfidfVectorizer` with Laplace-smoothed `MultinomialNB(alpha=0.1)`, fitted strictly on training data.
5. **Rigorous Empirical Evaluation:** Evaluate test metrics exclusively on unseen emails, calculating Accuracy, Spam Precision, Spam Recall, F1-Score, and saving confusion matrix artifacts.
6. **Production Streamlit Dashboard:** Design a responsive blue, white, and navy web application supporting direct text input, quick presets, `.eml` file uploading, dynamic probability estimation, and heuristic indicator analysis.

---

## 4. Tools and Technologies

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.14 (compatible with 3.10+) | Execution environment |
| **Email Processing** | Python `email` + `BeautifulSoup4` | MIME tree traversal, RFC 2047 header decoding, HTML stripping |
| **Feature Extraction** | Scikit-learn `TfidfVectorizer` | Sublinear term frequency, unigram/bigram extraction, stop words removal |
| **Classifier** | Scikit-learn `MultinomialNB` | Probabilistic classification with Laplace smoothing ($\alpha = 0.1$) |
| **Data Manipulation** | Pandas & NumPy | Tabular structuring, deduplication, stratified partitioning |
| **Visualization & Storage** | Matplotlib, Seaborn, Joblib | Confusion matrix heatmap generation and model serialization |
| **Web Dashboard** | Streamlit | Responsive user interface running at `http://localhost:8501` |

---

## 5. Dataset Source and Licensing Notes

- **Dataset:** SpamAssassin Public Email Corpus
- **Provider:** The Apache Software Foundation
- **Archive Files Used:**
  - `20030228_easy_ham.tar.bz2`: 2,500 legitimate personal and developer mailing list emails.
  - `20030228_spam.tar.bz2`: 500 unsolicited commercial, phishing, and scam emails.
- **Repository URL:** [https://spamassassin.apache.org/old/publiccorpus/](https://spamassassin.apache.org/old/publiccorpus/)
- **License / Terms:** Public domain / Open corpus distributed for academic and research evaluation.
- **Cleaned Data Summary:**
  - Raw parsed files: 3,000 emails
  - Duplicate emails removed: 46
  - Cleaned corpus total: **2,954 unique emails** (2,472 Ham [83.68%], 482 Spam [16.32%])

---

## 6. Architecture and Workflow

```
[SpamAssassin Archives]
          |
          v
[email_parser.py] ---> RFC 2047 Header Decoding & MIME Body Traversal
          |
          v
[Data Preprocessing] -> HTML Stripping + Deduplication (2,954 emails)
          |
          v
[Stratified Split] ----> 80% Train (2,363 emails)  |  20% Held-Out Test (591 emails)
                               |                                 |
                               v                                 |
                     [Fit TF-IDF (1-2 ngrams)]                   |
                               |                                 |
                     [Train MultinomialNB (alpha=0.1)]           |
                               |                                 |
                     [Save Pipeline to models/]                  |
                               |                                 v
                               +--------------------> [Evaluate Unseen Test Data]
                                                                 |
                                                                 v
                                                     [Outputs: JSON, CSV, PNG]
                                                                 |
                                                                 v
                                                     [Streamlit App (app.py)]
```

### Mathematical Formulation:
1. **TF-IDF Calculation:**
   $$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{if count} > 0 \text{ (sublinear)}$$
   $$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
   $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$
2. **Multinomial Naive Bayes Posterior Probability:**
   $$P(\text{Spam} \mid \mathbf{x}) \propto P(\text{Spam}) \prod_{i=1}^{V} P(x_i \mid \text{Spam})^{x_i}$$
   With Laplace smoothing ($\alpha = 0.1$):
   $$\hat{P}(x_i \mid c) = \frac{N_{ci} + 0.1}{N_c + 0.1 |V|}$$

---

## 7. Actual Measured Evaluation Results

Evaluated strictly on the held-out test split of **591 unseen emails**:

### 7.1 Empirical Metrics Table
| Metric | Score (Percentage) | Score (Decimal) | Evaluation Note |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **98.48%** | 0.9848 | 582 out of 591 unseen emails classified correctly |
| **Spam Precision** | **100.00%** | 1.0000 | **0 false positives** recorded across all 495 legitimate test emails |
| **Spam Recall** | **90.62%** | 0.9062 | Successfully intercepted 87 out of 96 spam emails |
| **Spam F1-Score** | **95.08%** | 0.9508 | Harmonic mean reflecting stellar reliability |

### 7.2 Confusion Matrix (Test Split: $n = 591$)
| | Predicted Ham (0) | Predicted Spam (1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Ham** | **495** ($TN$) | **0** ($FP$) | 495 |
| **Actual Spam** | **9** ($FN$) | **87** ($TP$) | 96 |
| **Total Predicted** | 504 | 87 | 591 |

---

## 8. Dashboard Features & Verification

The Streamlit interface at `http://localhost:8501` includes:
- **Dedicated Email Subject and Body Text Areas:** Accepts both fields and combines them for holistic analysis.
- **Direct `.eml` Upload Functionality:** Uses `email_parser.py` to parse real email exports from Outlook, Thunderbird, or Gmail.
- **Probabilistic Risk Indicator:** Displays real-time confidence percentages for Ham vs. Spam.
- **Heuristic Indicators:** Evaluates trigger phrases, link frequency, exclamation marks, and capitalization.
- **One-Click Quick Presets:** Pre-loads realistic phishing, prize claim, and corporate update emails for live demonstration.
- **Empty & Malformed Input Handling:** Provides clear user-facing warnings on empty submissions or invalid files.

---

## 9. Screenshot Insertion Checklist

Insert real screenshots of the working system into the final PDF/document submission:

- [ ] **Figure 1: Training Terminal Output**  
  *PowerShell console displaying `python train_model.py` parsing 3,000 email files, deduplication, and the final classification report.*
- [ ] **Figure 2: Streamlit Dashboard Overview**  
  *Browser view of `http://localhost:8501` showcasing the blue-and-white header, live metric cards (98.48% Accuracy, 100% Precision), and dataset breakdown.*
- [ ] **Figure 3: Confusion Matrix Heatmap**  
  *Expanded view showing `outputs/confusion_matrix.png` rendered inside the dashboard.*
- [ ] **Figure 4: Phishing/Spam Analysis Result**  
  *Result card showing `🚨 SPAM DETECTED` with 78%+ Spam probability and identified trigger keywords.*
- [ ] **Figure 5: Legitimate Business Email Result**  
  *Result card showing `✅ NOT SPAM (HAM)` with 98%+ Ham probability on a workplace update.*
- [ ] **Figure 6: .eml File Upload & Heuristics**  
  *Uploader displaying auto-extracted subject and body from an uploaded `.eml` message.*

---

## 10. Conclusion and Limitations

### 10.1 Key Achievements
- Replaced SMS text data with real-world email archives from the Apache SpamAssassin corpus.
- Built a leak-free Scikit-learn Pipeline achieving **98.48% accuracy** and a **100.00% spam precision rate** ($0$ false alarms).
- Developed a full-featured web dashboard with `.eml` support and heuristic analysis.

### 10.2 Limitations & Future Work
- **Header Authentication:** Expanding analysis to inspect DKIM, SPF, and DMARC DNS verification records.
- **Attachment Signatures:** Scanning file extensions and hashes of email attachments for known malware patterns.
- **Adversarial Defenses:** Incorporating subword embeddings (e.g. Byte-Pair Encoding) to resist character-substitution obfuscation.
