import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score 
import matplotlib.pyplot as plt

# Sample data for model comparison
data = {
    'StudyHours': [2, 3, 5, 7, 1, 4, 6, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0,0,0,0,0,1,1,1,1,1]
}

# Create a DataFrame
df = pd.DataFrame(data)
print(df.head())

# Split data into features and target variable
X = df[['StudyHours', 'PrevExamScore']] # Features
y = df['Pass'] # Target variable (0 = Fail, 1 = Pass)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, logistic_predictions)
print(f"Logistic Regression Accuracy: {accuracy_logistic}")

# Calculate metrics
accuracy = accuracy_score(y_test, logistic_predictions)
precision = precision_score(y_test, logistic_predictions)
recall = recall_score(y_test, logistic_predictions)
f1 = f1_score(y_test, logistic_predictions)

# Compare and Evaluate Models
print(f"Accuracy: {accuracy_logistic}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# K-fold Cross-validation 
cv_scores = cross_val_score(logistic_model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
print(f"Cross-Validation accuracies: {cv_scores}")
print(f'Mean cross-validation accuracy: {np.mean(cv_scores)}')

# Cross-validation with multiple metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(logistic_model, X, y, cv=5, scoring=scoring)

print(f"Cross-validation Accuracy: {np.mean(cv_results['test_accuracy'])}")
print(f"Cross-validation Precision: {np.mean(cv_results['test_precision'])}")
print(f"Cross-validation Recall: {np.mean(cv_results['test_recall'])}")
print(f"Cross-validation F1-Score: {np.mean(cv_results['test_f1'])}")
