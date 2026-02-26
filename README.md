# 🍷 Wine Varietal Identification — ML Classification Benchmark

> Comparing six machine learning algorithms to classify wine types from physicochemical properties, with rigorous cross-validation and hyperparameter tuning.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

---

## 📋 Overview

A comprehensive benchmark study evaluating six classification algorithms for wine varietal identification using physicochemical properties. The project follows a full ML workflow — EDA, preprocessing, training, hyperparameter tuning, and multi-metric evaluation — to determine which algorithm best suits this classification problem.

**Best performers:** Random Forest & SVM — achieving **98–99% accuracy** on test data.

---

## 🧪 Algorithms Compared

| Algorithm | Type | Notes |
|---|---|---|
| Logistic Regression | Linear | Baseline classifier |
| Decision Tree | Non-linear | Interpretable rules |
| Random Forest | Ensemble | Best overall performance |
| K-Nearest Neighbors | Instance-based | Distance-based learning |
| Naive Bayes | Probabilistic | Bayes' theorem based |
| Support Vector Machine | Maximum margin | Best with RBF kernel |

---

## 📊 Results

| Model | Accuracy | Highlights |
|---|---|---|
| **Random Forest** | ~99% | Best interpretability + performance |
| **SVM (RBF kernel)** | ~98% | Best at nonlinear decision boundaries |
| Decision Tree | High | Good interpretability |
| Logistic Regression | Good | Strong baseline |
| KNN | Good | Simple, effective |
| Naive Bayes | Moderate | Fast, probabilistic |

---

## 🔬 Methodology

**Data Exploration & Preprocessing**
Extensive EDA covering feature distributions, correlations, and outlier detection. Applied normalization and standardization across all features before model training.

**Model Evaluation**
All models were assessed using 10-fold cross-validation and four metrics — accuracy, precision, recall, and F1-score — alongside confusion matrices to capture a complete picture of performance.

**Hyperparameter Tuning**
Each model was optimized with GridSearchCV to find the best hyperparameter combination while guarding against overfitting.

**Feature Importance Analysis**
The three most predictive features identified were alcohol content, flavonoid levels, and color intensity — consistent with oenological domain knowledge.

---

## 🔑 Key Findings

- Tree-based methods offer strong performance with the added benefit of feature importance interpretation
- SVM with RBF kernel excels at capturing complex nonlinear relationships
- All six models significantly outperform random baseline, validating the predictive signal in physicochemical features
- Alcohol content, flavonoids, and color intensity are the strongest wine varietal predictors

---

## 📁 Project Structure

```
.
├── notebooks/
│   └── wine_classification.ipynb   # Full analysis and model comparison
├── data/
│   └── wine.csv                    # Wine dataset
├── visuals/
│   ├── correlation_matrix.png
│   ├── feature_distributions.png
│   └── model_comparison.png
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| ML | Scikit-Learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |
| Version Control | Git, GitHub |

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/wine-classification.git
cd wine-classification
pip install -r requirements.txt
jupyter notebook notebooks/wine_classification.ipynb
```

---

## 🍾 Business Implications

This framework provides a reusable blueprint for selecting optimal classifiers in agricultural and food science applications — with natural extensions to other quality classification problems across the food and beverage industry.

---

## 📄 License

This project is licensed under the MIT License.