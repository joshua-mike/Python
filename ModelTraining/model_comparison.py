import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

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

# Decision Tree Model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
decision_tree_predictions = decision_tree_model.predict(X_test)
accuracy_decision_tree = accuracy_score(y_test, decision_tree_predictions)
print(f"Decision Tree Accuracy: {accuracy_decision_tree}")

# Compare and Evaluate Models
print("Logistic Regression:")
print(f"Accuracy: {accuracy_logistic}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))
print("Classification Report:")
print(classification_report(y_test, logistic_predictions))

print("Decision Tree:")
print(f"Accuracy: {accuracy_decision_tree}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, decision_tree_predictions))
print("Classification Report:")
print(classification_report(y_test, decision_tree_predictions))

# Visualize Decision Tree
plt.figure(figsize=(12,8))
tree.plot_tree(decision_tree_model, feature_names=['StudyHours', 'PrevExamScore'], class_names=['Fail', 'Pass'], filled=True)
plt.title("Decision Tree for Classifying Pass/Fail Based on Study Hours and Previous Exam Score")
plt.show()