# 🧠 Mental Health Text Classification using Machine Learning, Deep Learning, and Transformers

> End-to-End Natural Language Processing project for multi-class mental health text classification using Machine Learning, Deep Learning, and Transformer-based models with FastAPI deployment.

---

# Project Overview

Mental health disorders have become one of the most important public health challenges worldwide. Automatically identifying mental health conditions from textual data can support early intervention and assist healthcare professionals.

This project presents a complete Natural Language Processing pipeline for multi-class mental health text classification. Different Machine Learning, Deep Learning, and Transformer-based approaches were implemented and compared to identify the best-performing model.

The final selected model is **Microsoft DeBERTa V3**, which achieved the highest classification performance and was deployed using **FastAPI** for real-world inference.

---

# Project Pipeline

```text
Raw Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Text Cleaning & Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Machine Learning Models
      │
      ▼
Deep Learning Models
      │
      ▼
Transformer Fine-Tuning
      │
      ▼
Model Comparison
      │
      ▼
Best Model Selection
      │
      ▼
FastAPI Deployment
      │
      ▼
Frontend Integration
```

---

# Project Structure

```text
Mental_Health_Classification/

│

├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Machine_Learning.ipynb
│   ├── 05_Deep_Learning.ipynb
│   ├── 06_DeBERTa.ipynb
│   ├── 07_Model_Comparison.ipynb
│   └── 08_Deployment.ipynb
│
├── deployment/
│
├── models/
│
├── tokenizers/
│
├── encoders/
│
├── utils/
│
├── results/
│
├── figures/
│
└── README.md
```

---

# Dataset

The project uses a multi-class mental health dataset consisting of textual statements collected from online mental health discussions.

The dataset contains multiple psychological categories including depression, anxiety, stress, and other mental health conditions.

**Dataset Characteristics**

- Multi-class classification
- English language
- Text-based samples
- Imbalanced class distribution

---

# Text Preprocessing

Two preprocessing pipelines were implemented.

## Classical Machine Learning

- HTML removal
- URL removal
- Email removal
- Mention removal
- Lowercasing
- Punctuation removal
- Number removal
- Lemmatization

## Transformer Models

Only lightweight preprocessing was applied:

- HTML removal
- URL removal
- Email removal
- Mention removal

No aggressive normalization was applied to preserve the contextual information required by Transformer models.

---

# Feature Engineering

Different feature extraction techniques were investigated.

## Machine Learning

- TF-IDF Vectorization

## Deep Learning

- Tokenization
- Vocabulary Construction
- Sequence Padding

## Transformers

- HuggingFace AutoTokenizer
- Dynamic Attention Mask
- Token IDs

---

# Implemented Models

## Machine Learning

- Logistic Regression
- Optimized Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- Random Forest
- XGBoost

---

## Deep Learning

- Baseline LSTM
- Bidirectional LSTM

---

## Transformer

- Microsoft DeBERTa V3

---

# Model Performance

| Rank | Model | Accuracy | Weighted F1 |
|------|------|----------|-------------|
| 🥇 | DeBERTa V3 | 84.04% | 84.01% |
| 🥈 | Bidirectional LSTM | 76.76% | 77.15% |
| 🥉 | Optimized Logistic Regression | 76.63% | 76.68% |
| 4 | Logistic Regression | 76.96% | 76.33% |
| 5 | XGBoost | 76.62% | 76.30% |
| 6 | Linear SVM | 76.16% | 75.69% |
| 7 | Baseline LSTM | 74.77% | 75.43% |
| 8 | Multinomial Naive Bayes | 70.28% | 68.72% |
| 9 | Random Forest | 68.48% | 65.70% |

---

# Why DeBERTa V3?

The Transformer model achieved the best overall performance while demonstrating superior contextual understanding of mental health related statements.

Advantages:

- Highest Accuracy
- Highest Weighted F1
- Highest Macro F1
- Better contextual understanding
- Better generalization

---

# Deployment Architecture

```text
User
 │
 ▼
FastAPI
 │
 ▼
Text Preprocessing
 │
 ▼
Tokenizer
 │
 ▼
DeBERTa V3
 │
 ▼
Softmax
 │
 ▼
Top-3 Predictions
 │
 ▼
Frontend
```

---

# Example Prediction

Input

```text
I feel hopeless and empty.
Nothing makes me happy anymore.
```

Output

```json
{
    "prediction":"Depression",
    "confidence":94.82,

    "top3":[

        {
            "rank":1,
            "label":"Depression",
            "confidence":94.82
        },

        {
            "rank":2,
            "label":"Anxiety",
            "confidence":3.74
        },

        {
            "rank":3,
            "label":"Stress",
            "confidence":1.44
        }

    ]
}
```

---

# API Endpoints

## Home

```http
GET /
```

Returns API status.

---

## Prediction

```http
POST /predict
```

Example Request

```json
{
    "text":"I feel hopeless and empty."
}
```


---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- PyTorch
- HuggingFace Transformers
- FastAPI
- Joblib
- Uvicorn

---

# Results

## Model Comparison Dashboard

> **Insert Dashboard Screenshot Here**

---

## Confusion Matrix

> **Insert Confusion Matrix Image Here**

---

## Learning Curves

> **Insert Learning Curves Here**

---

## FastAPI Swagger

> **Insert Swagger Screenshot Here**

---

## Frontend Integration

> **Insert Frontend Screenshot Here**

---

# Future Improvements

- Azure Cloud Deployment
- Docker Containerization
- CI/CD Pipeline
- Explainable AI (SHAP / LIME)
- Model Monitoring
- Multilingual Support
- Continuous Training Pipeline

---

# Author

**Menna Talla Gamal**

Artificial Intelligence Engineer

