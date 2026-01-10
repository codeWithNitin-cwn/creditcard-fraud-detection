
# Credit Card Fraud Detection Project

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE

# Step 2: Load Dataset
df = pd.read_csv('creditcard.csv')
print("Dataset loaded successfully.")

# Step 3: Basic Info
print(df.info())
print(df['Class'].value_counts())

# Step 4: Correlation Analysis
plt.figure(figsize=(20, 15))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Matrix")
plt.show()

threshold = 0.9
corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
high_corr = [(a, b) for (a, b) in corr_pairs.index if a != b and corr_matrix.loc[a, b] > threshold]

print("Highly correlated feature pairs (corr > 0.9):")
for a, b in high_corr:
    print(f"{a} - {b}: {corr_matrix.loc[a, b]:.2f}")

# Step 5: Amount-Based Fraud Pattern
plt.figure(figsize=(10, 5))
sns.histplot(data=df[df['Class'] == 1], x='Amount', bins=100, color='red', label='Fraud', kde=True)
sns.histplot(data=df[df['Class'] == 0], x='Amount', bins=100, color='green', label='Legit', kde=True)
plt.legend()
plt.title("Distribution of Transaction Amounts (Fraud vs Legit)")
plt.xlabel("Transaction Amount")
plt.ylabel("Count")
plt.show()

top_fraud_amounts = df[df['Class'] == 1]['Amount'].round(2).value_counts().head(10)
print("\nTop 10 transaction amounts in fraud cases:")
print(top_fraud_amounts)

# Step 6: Time Series Analysis
df['Hour'] = (df['Time'] / 3600).apply(np.floor) % 24

plt.figure(figsize=(10, 5))
sns.histplot(df[df['Class'] == 1]['Hour'], bins=24, color='red', label='Fraud', kde=False)
sns.histplot(df[df['Class'] == 0]['Hour'], bins=24, color='green', label='Legit', kde=False)
plt.title("Transactions per Hour (Fraud vs Legit)")
plt.xlabel("Hour of Day")
plt.ylabel("Transaction Count")
plt.legend()
plt.show()

# Step 7: Preprocessing
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])
X = df.drop(['Class'], axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE class distribution:")
print(pd.Series(y_train_smote).value_counts())

# Step 9: Train Random Forest
print("Training Random Forest... please wait ⏳")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_smote, y_train_smote)
print("✅ Random Forest training completed.")

# Step 10: Evaluation
y_pred = rf.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_pred))
