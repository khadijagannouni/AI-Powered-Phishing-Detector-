# AI-Powered Phishing Email Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/LIME%20%26%20SHAP-Explainability-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge"/>
</p>

<p align="center">
  An end-to-end AI system that classifies emails as <strong>Phishing</strong> or <strong>Legitimate</strong>,
  explains every decision with LIME & SHAP, and stress-tests itself by generating synthetic phishing attacks via LLM.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dataset](#dataset)
- [Team](#team)

---

## Overview

Phishing attacks are among the most prevalent cybersecurity threats today — and attackers are increasingly using AI to craft more convincing emails. This project fights back with AI.

We built a supervised Machine Learning pipeline that:
- **Classifies** incoming emails as *Phish* or *Ham* using NLP feature extraction and a trained classifier
- **Explains** its decisions token by token using LIME and SHAP so users understand *why* an email was flagged
- **Simulates** novel phishing attacks using a Large Language Model (LLM) to probe the model for blind spots
- **Visualises** all results through an interactive Streamlit web dashboard

> **Course:** Cybersecurity  
> **Supervisor:** Dr. Manel Abdelkader  

---

## Features

| Feature | Description |
|---|---|
| **Email Classification** | Classifies any email as Phish or Ham with a confidence score |
| **LIME Explainability** | Highlights the exact words that drove the model's decision |
| **SHAP Analysis** | Shows global feature importance across the full dataset |
| **Attack Simulation** | Generates realistic synthetic phishing emails via Claude / OpenAI API |
| **Analytics Dashboard** | Confusion matrix, ROC curve, confidence distributions, session history |
| **Feedback Loop** | Users can submit corrected labels for future model retraining |

---

## System Architecture

```
+------------------------------------------------------------------+
|                          DATA LAYER                              |
|        Kaggle Phishing Dataset · pandas · train/test split       |
+-----------------------------+------------------------------------+
                              |
+-----------------------------v------------------------------------+
|                        NLP & ML CORE                             |
|   NLTK preprocessing · TF-IDF vectorizer · URL & urgency feats   |
|           Logistic Regression  <-->  Random Forest               |
+---------------+----------------------------------+---------------+
                |                                  |
+---------------v----------------+   +-------------v--------------+
|    EXPLAINABILITY MODULE        |   |      ATTACK SIMULATION     |
|    LIME · SHAP                  |   |  Claude API · OpenAI API   |
|    Local + Global XAI           |   |  Prompt templates          |
+---------------+----------------+   +-------------+--------------+
                |                                  |
+---------------v----------------------------------v---------------+
|                    FRONTEND & OUTPUT LAYER                       |
|          Streamlit · Plotly · matplotlib · GitHub                |
+------------------------------------------------------------------+
```

**Data flows sequentially** through four layers:
1. Raw emails are loaded and split into train / test / feedback sets
2. Text is preprocessed and transformed into numerical feature vectors
3. The classifier predicts a label and confidence score
4. LIME/SHAP explanations and Plotly charts are rendered on the dashboard

---

## Project Structure

```
AI-Powered-Phishing-Detector/
|
+-- src/                            <- Core Python modules
|   +-- __init__.py
|   +-- email_processor.py          <- Text cleaning, tokenization, lemmatization (NLTK)
|   +-- feature_extractor.py        <- TF-IDF vectorizer + URL count + urgency score
|   +-- phishing_classifier.py      <- Logistic Regression & Random Forest classifiers
|   +-- explainability_module.py    <- LIME (local) and SHAP (global) explanations
|   +-- attack_simulator.py         <- LLM-powered synthetic phishing email generator
|   +-- evaluation_report.py        <- Metrics aggregation, CSV export
|
+-- data/                           <- Dataset directory (not tracked by git)
|   +-- phishing_dataset.csv
|
+-- models/                         <- Saved model artifacts (not tracked by git)
|   +-- classifier.pkl
|   +-- feature_extractor.pkl
|   +-- tfidf_vectorizer.pkl
|
+-- app.py                          <- Streamlit web dashboard (entry point)
+-- train.py                        <- Model training & evaluation pipeline
+-- requirements.txt                <- Python dependencies
+-- .gitignore
+-- README.md
```

---

## Tech Stack

| Layer | Tool / Library | Purpose |
|---|---|---|
| Language | Python 3.11 | Primary language |
| Data | pandas | Dataset loading and manipulation |
| Data | scikit-learn `train_test_split` | Reproducible stratified splits |
| NLP | NLTK | Tokenization, stop-word removal, lemmatization |
| NLP | spaCy | Named entity recognition (sender, org, URL detection) |
| ML | scikit-learn `TfidfVectorizer` | Text to numerical feature vectors |
| ML | Logistic Regression | Fast, interpretable baseline classifier |
| ML | Random Forest | Ensemble classifier for comparison |
| ML | scikit-learn metrics | Accuracy, F1, precision, recall, confusion matrix |
| Persistence | joblib | Save and load trained model artifacts |
| Explainability | LIME | Local per-email token-level explanations |
| Explainability | SHAP | Global Shapley value feature importance |
| Attack Simulation | Claude API (Anthropic) | Primary LLM for synthetic email generation |
| Attack Simulation | OpenAI API | Fallback LLM provider |
| Attack Simulation | requests | HTTP client for LLM API calls |
| Frontend | Streamlit | Interactive web application |
| Frontend | Plotly | Interactive charts and visualisations |
| Frontend | matplotlib | Static confusion matrix and training curve figures |
| DevOps | Git + GitHub | Version control and collaboration |

---

## Getting Started

### Prerequisites
- Python **3.11** — [download here](https://python.org/downloads)
- Git — [download here](https://git-scm.com)

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Powered-Phishing-Detector.git
cd AI-Powered-Phishing-Detector
```

### 2 — Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Download required language data

```bash
# NLTK corpora
python -m nltk.downloader punkt stopwords wordnet punkt_tab

# spaCy English model
python -m spacy download en_core_web_sm
```

### 5 — Add the dataset

Download the dataset from Kaggle and place it in the `data/` folder:

```
data/
+-- phishing_dataset.csv
```

> Dataset: [Phishing Email Dataset — Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

### 6 — Train the model

```bash
mkdir models
python train.py --data data/phishing_dataset.csv --model logistic_regression
```

> You can also use `--model random_forest` to train the ensemble model instead.  
> Trained artifacts are saved to `models/`.

### 7 — Launch the dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Usage

### Classify an email
Navigate to **Classify Email** in the sidebar, paste an email subject and body, and click **Analyse**.

### View explanations
Navigate to **Explainability** to run LIME on any email and see which words pushed the model toward Phish or Ham.

### Simulate attacks
Navigate to **Attack Simulation**, select an attack type (e.g. spear phishing, whaling), and generate synthetic phishing variants to test the model.

### View analytics
Navigate to **Analytics Dashboard** to see session statistics, confidence distributions, and confusion matrix.

---

### Environment Variables — Attack Simulation only

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

> Never commit your `.env` file. It is already excluded in `.gitignore`.  
> The classifier and dashboard work fully without these keys.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) |
| Format | CSV |
| Labels | `1` = Phishing · `0` = Legitimate (Ham) |
| Split | 75% train · 20% test · 5% feedback |

---

## Team

| Name | Role |
|---|---|
| **Khadija Gannouni** | ML Pipeline · Feature Extraction · Attack Simulation |
| **Maram Abdallah** | Explainability Module · Dashboard · Evaluation |

> Supervised by **Dr. Manel Abdelkader**

---

## License

This project was developed for academic purposes as part of a university cybersecurity course.  
It is not licensed for commercial use.

---

<p align="center">Developed by Khadija Gannouni and Maram Abdallah</p>
