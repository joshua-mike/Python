import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Sample dataset - binary classification based on exam scores
data = {
    'StudyHours': [34, 78, 50, 90, 60, 85, 70, 95],
    'Pass': [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
print(df.head())

# Split the dataset into features and target variable
X = df[['StudyHours']]  # Features
y = df['Pass']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# Train the logistic regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Display the model's learned coefficients and intercept
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_[0]}")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Display the predictions alongside the actual values
print("Predicted Pass/Fail:", y_pred)
print("Actual Pass/Fail:", y_test.values)

# Evaluate the model - calculate accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualize the results by plotting the decision boundary along with the data points
study_hour_range = np.linspace(X.min(), X.max(), 100)

y_prob = model.predict_proba(study_hour_range.reshape(-1, 1))[:, 1]
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(study_hour_range, y_prob, color='red', linewidth=2, label='Logisitic Regression Curve')

# Add labels and title to the plot
plt.xlabel('Study Hours')
plt.ylabel('Probablity of Passing')
plt.title('Logistic Regression: Study Hours vs. Probability of Passing')
plt.legend()
plt.show()