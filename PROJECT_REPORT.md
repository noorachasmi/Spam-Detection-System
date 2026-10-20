# PROJECT REPORT
# Spam Email Detection System

**Author / Intern:** Machine Learning Engineering Intern  
**Project Track:** Natural Language Processing & Applied Machine Learning  
**Workspace:** `Spam Detection`  
**Date:** October 2026  

---

## 1. Project Title
**Spam Email Detection System Using TF-IDF Vectorization and Multinomial Naive Bayes**

---

## 2. Problem Statement
Digital communication via email and short messaging services (SMS) has become ubiquitous for professional, educational, and personal discourse. Concurrently, malicious actors exploit these channels to deploy unsolicited communications, fraudulent phishing scams, financial extortion, and malware distribution. 

Manual filtering of such communications is infeasible given the scale of daily traffic. An automated, real-time message classification system is necessary to identify and sequester malicious content while ensuring legitimate communications are delivered without interruption. Crucially, the system must balance high detection recall with minimal false positives, as wrongly misclassifying a legitimate business or transactional communication as spam can have severe consequences.

---

## 3. Project Objectives
1. **Automated Data Acquisition:** Programmatically acquire and validate the benchmark UCI SMS Spam Collection dataset.
2. **Robust Data Preprocessing:** Clean and deduplicate the raw corpus to eliminate redundancy and prevent data contamination across train and test partitions.
3. **Leak-Free Machine Learning Pipeline:** Implement a Scikit-learn pipeline coupling `TfidfVectorizer` and `MultinomialNB`, ensuring the vectorizer is fitted strictly on training data.
4. **Rigorous Empirical Evaluation:** Evaluate performance strictly on an unseen 20% test partition using standard classification metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
5. **Interactive Web Interface:** Build a responsive, user-friendly Streamlit web application running at `localhost:8501` that loads genuine evaluation metrics dynamically and provides real-time probabilistic message classification with input validation.
6. **Academic & Ethical Transparency:** Clearly articulate dataset characteristics, limitations, and domain boundaries (SMS vs. Email) to ensure the project does not overstate its real-world performance.

---

## 4. Tools and Technologies

| Technology / Tool | Version / Specification | Role in Project |
| :--- | :--- | :--- |
| **Python** | 3.14 / 3.10+ | Core programming runtime |
| **Pandas** | 3.0+ | Tabular data manipulation, exploration, and deduplication |
| **Scikit-learn** | 1.9+ | TF-IDF feature extraction, Naive Bayes modeling, and evaluation |
| **Joblib** | 1.6+ | Serialization and deserialization of the trained Scikit-learn pipeline |
| **Streamlit** | 1.66+ | Web application framework for the user interface |
| **Git & GitHub** | Standard VCS | Version control and reproducibility |

---

## 5. Dataset Description and Source

- **Name:** SMS Spam Collection Dataset
- **Repository:** Official UCI Machine Learning Repository
- **Dataset ID:** 228
- **URL:** [https://archive.ics.uci.edu/dataset/228/sms+spam+collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Format:** Tab-separated values (`label \t message`)
- **Initial Size:** 5,574 text entries (4,827 Ham, 747 Spam)
- **Cleaned Size (after deduplication):** 5,171 text entries (4,518 Ham, 653 Spam)
- **Class Proportions:** 87.37% Ham (Legitimate) / 12.63% Spam

> **Crucial Domain Clarification:**  
> The corpus contains real-world SMS communications collected for mobile phone spam research. While core spam characteristics (monetary enticements, artificial urgency, deceptive URLs) translate well to email text, SMS messages lack email-specific attributes such as SPF/DKIM headers, multi-part MIME boundaries, HTML tables, and attachment binary streams. The system evaluates text content patterns; predictions on full email payloads represent an approximation of textual content filtering.

---

## 6. System Architecture and Algorithm

### 6.1 System Flow
```
[Raw TSV File] -> [Data Validation & Deduplication] -> [Stratified Train-Test Split (80/20)]
                                                                 |
               +-------------------------------------------------+
               |                                                 |
         [X_train, y_train]                                [X_test, y_test]
               |                                                 |
     [TF-IDF Fit & Transform]                                    |
               |                                                 |
    [Multinomial NB Training]                                    |
               |                                                 |
   [Joblib Pipeline Export]                                      |
               |                                                 |
               +--------------> [Pipeline Predict] <-------------+
                                       |
                       [Generate Test Set Metrics & CM]
                                       |
                     [Save metrics.json & confusion_matrix.csv]
                                       |
                             [Streamlit Web App]
```

### 6.2 Mathematical Formulation of Algorithms

#### 1. Term Frequency-Inverse Document Frequency (TF-IDF)
For term $t$ in document $d$ within a corpus of size $N$:
$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
$$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$
Features are normalized using Euclidean ($L_2$) norm. We extract both unigrams and bigrams (`ngram_range=(1, 2)`) to capture multi-word cues like *"cash prize"* or *"click here"*.

#### 2. Multinomial Naive Bayes Classifier
Based on Bayes' Theorem under the conditional independence assumption:
$$P(c \mid \mathbf{x}) \propto P(c) \prod_{i=1}^{n} P(x_i \mid c)$$
To prevent zero probabilities for unobserved vocabulary terms, Laplace smoothing ($\alpha = 0.1$) is applied:
$$\hat{P}(x_i \mid c) = \frac{N_{ci} + \alpha}{N_c + \alpha |V|}$$
where $N_{ci}$ is the count of term $i$ in class $c$, $N_c$ is the total count of all features in class $c$, and $|V|$ is the vocabulary size.

---

## 7. Implementation Steps

1. **Step 1: Environment & Directory Setup**  
   Configured `.gitignore`, declared pinned dependencies in `requirements.txt`, and generated directories `data/`, `models/`, and `outputs/`.
2. **Step 2: Automated Dataset Download (`download_data.py`)**  
   Built an automated Python script leveraging `urllib.request` and `zipfile` to fetch `sms+spam+collection.zip` from the UCI repository and extract `SMSSpamCollection`.
3. **Step 3: Data Cleaning & Validation (`train_model.py`)**  
   Read tab-separated records, checked null values (0 found), and identified 403 duplicate messages. Removed duplicates, yielding 5,171 pristine records.
4. **Step 4: Stratified Splitting**  
   Executed `train_test_split(..., test_size=0.20, random_state=42, stratify=y)` to produce 4,136 training samples and 1,035 testing samples, maintaining identical class distributions across both splits.
5. **Step 5: Pipeline Assembly and Leak-Free Training**  
   Created a single Scikit-learn `Pipeline` chaining `TfidfVectorizer` and `MultinomialNB(alpha=0.1)`. Called `.fit()` exclusively on the training subset.
6. **Step 6: Rigorous Testing & Artifact Persistence**  
   Evaluated predictions on the unseen test set. Serialized the pipeline to `models/spam_classifier_pipeline.joblib`, wrote `outputs/evaluation_metrics.json`, and stored `outputs/confusion_matrix.csv`.
7. **Step 7: Web Application Development (`app.py`)**  
   Constructed an interactive Streamlit dashboard featuring custom CSS cards, real-time metrics loading, input validation, confidence bars, and quick-test message presets.

---

## 8. Actual Empirical Results

All reported metrics were obtained exclusively from the unseen 20% test subset ($n = 1,035$):

### 8.1 Performance Metrics Table
| Metric | Score (Decimal) | Score (Percentage) |
| :--- | :---: | :---: |
| **Overall Accuracy** | 0.9826 | **98.26%** |
| **Spam Precision** | 0.9748 | **97.48%** |
| **Spam Recall** | 0.8855 | **88.55%** |
| **Spam F1-Score** | 0.9280 | **92.80%** |

### 8.2 Confusion Matrix (Test Set: $n = 1,035$)
| | Predicted Ham (0) | Predicted Spam (1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Ham** | **901** ($TN$) | **3** ($FP$) | 904 |
| **Actual Spam** | **15** ($FN$) | **116** ($TP$) | 131 |
| **Total Predicted** | 916 | 119 | 1,035 |

### 8.3 Classification Report Breakdown
```
              precision    recall  f1-score   support
     Ham (0)       0.98      1.00      0.99       904
    Spam (1)       0.97      0.89      0.93       131

    accuracy                           0.98      1035
   macro avg       0.98      0.94      0.96      1035
weighted avg       0.98      0.98      0.98      1035
```

---

## 9. Genuine Screenshot Insertion Guide

When assembling your final PDF submission or presentation slides, take authentic screenshots of the running system and place them in the following designated sections:

### Screenshot 1: Dataset Acquisition and Model Training Terminal Output
- **What to capture:** The PowerShell terminal showing execution of `python train_model.py`, including dataset record counts, deduplication results, and the printed classification report.
- **Placement:** Insert directly below Section 7 (Implementation Steps).
- **Caption:** *Figure 1: Terminal log of dataset preprocessing, stratified train/test split, and model training.*

> `[INSERT SCREENSHOT: Terminal execution showing train_model.py output]`

---

### Screenshot 2: Streamlit Dashboard Overview & Dynamic Performance Cards
- **What to capture:** The web browser displaying `http://localhost:8501`, showcasing the blue-and-white header, the domain limitation notice, and the 5 metric cards (Test Accuracy: 98.26%, Precision: 97.48%, Recall: 88.55%, F1: 92.80%, Dataset Size: 5,171).
- **Placement:** Insert directly below Section 8 (Actual Empirical Results).
- **Caption:** *Figure 2: Streamlit web dashboard displaying dynamically loaded model evaluation metrics.*

> `[INSERT SCREENSHOT: Streamlit browser tab showing top metrics cards]`

---

### Screenshot 3: Interactive Spam Classification Output
- **What to capture:** The web interface after clicking "Predict" on a sample spam message (e.g., *"URGENT! You have won a 1000 GBP cash prize..."*), displaying the red "SPAM" card and the confidence progress bars.
- **Placement:** Insert in the System Evaluation section.
- **Caption:** *Figure 3: Web application detecting a spam message with probability breakdown.*

> `[INSERT SCREENSHOT: Streamlit prediction result showing Red SPAM alert and 99%+ probability]`

---

### Screenshot 4: Interactive Legitimate (Ham) Classification Output
- **What to capture:** The web interface after clicking "Predict" on a legitimate message (e.g., *"Hi team, please find the updated slides attached for tomorrow's 10 AM project review."*), displaying the green "NOT SPAM (HAM)" card.
- **Placement:** Insert in the System Evaluation section.
- **Caption:** *Figure 4: Web application correctly classifying a legitimate workplace communication.*

> `[INSERT SCREENSHOT: Streamlit prediction result showing Green NOT SPAM alert]`

---

### Screenshot 5: Input Validation Warning (Empty Message Handling)
- **What to capture:** The web interface when the "Predict" button is pressed with an empty text area, displaying the yellow warning message *"Input is empty! Please enter text before requesting a prediction."*
- **Placement:** Insert in the User Interface Features section.
- **Caption:** *Figure 5: User input validation preventing empty inference requests.*

> `[INSERT SCREENSHOT: Streamlit warning alert on empty input submission]`

---

## 10. Conclusion and Limitations

### 10.1 Key Achievements
- Engineered a lightweight, self-contained NLP text classification pipeline achieving **98.26% test accuracy** and **97.48% spam precision**.
- Successfully curbed the False Positive rate to **0.33%** (only 3 false alarms out of 904 legitimate test messages).
- Developed a professional, production-grade Streamlit web interface with dynamic data binding and robust error handling.
- Ensured zero data leakage through encapsulated Scikit-learn pipelines.

### 10.2 Limitations & Future Work
1. **SMS vs. Email Format Divergence:** The system was trained on mobile SMS data. It does not parse MIME headers, email sender reputations (SPF/DKIM/DMARC), attachments, or complex HTML formatting. Future versions should incorporate public email corpora such as the Enron Spam Dataset or Apache SpamAssassin.
2. **Adversarial Obfuscation:** Attackers frequently employ character leetspeak (e.g., `fr33`, `w!nner`), zero-width spaces, or image-based spam to bypass bag-of-words tokenizers. Integrating subword tokenization (such as Byte-Pair Encoding or WordPiece) and deep learning models (such as DistilBERT) would bolster adversarial robustness.
3. **Concept Drift:** Spammer tactics evolve over time. Deploying an active learning feedback loop where users can flag misclassifications would allow continuous model retraining.

---

## 11. Dataset Reference
- **Citation:** Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng '11), Mountain View, CA.
- **UCI Repository Record:** [https://archive.ics.uci.edu/dataset/228/sms+spam+collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
