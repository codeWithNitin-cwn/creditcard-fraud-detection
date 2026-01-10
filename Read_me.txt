#  Credit Card Fraud Detection

This project uses machine learning techniques to detect fraudulent credit card transactions. The dataset used is highly imbalanced, and special techniques like SMOTE and Random Forests are applied to improve detection.

---

##  Dataset

We use the **Credit Card Fraud Detection dataset** from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
Due to file size limits on GitHub, the dataset is **not included** in this repository.

 Please download it manually from:  
 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading:
1. Extract the ZIP file.
2. Move `creditcard.csv` into the root folder of this project (same location as your `main.py` or Jupyter notebook).

---

## Features Used

The dataset contains 31 features:
- Time, Amount — standard features
- V1 to V28 — anonymized PCA-transformed features
- Class — label (0 = Legit, 1 = Fraud)

---

## What This Project Does

### Data Preprocessing
- Standardization of `Time` and `Amount`
- Handling class imbalance using **SMOTE**


### Exploratory Analysis
- Correlation Matrix
- Fraud Amount Distribution
- Fraud Frequency Over Time (Hour of Day)


### Model Training
- Random Forest classifier trained on SMOTE-balanced data
- Evaluation using Confusion Matrix, Classification Report, ROC-AUC


### Live User Prediction
- Predicts if a transaction (with amount input) is legitimate or fraudulent using trained model


---

## Visualizations

The notebook includes:
- Correlation heatmaps
- Distribution plots for fraudulent vs. legitimate transactions
- Time series (hourly) fraud analysis

---


## Requirements

Make sure you have these Python packages installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
