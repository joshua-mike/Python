import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Sample dataset
# data = {
#    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
#    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
#}

data = {
    'StudyHours': [1, 7, 3, 9, 5, 6, 2, 8, 4, 10],
    'PrevExamScore': [30, 88, 45, 96, 60, 71, 32, 91, 62, 98],
    'Pass': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LASSO model with alpha regularization parameter
lasso_model = Lasso(alpha=0.1)

# Train the model
lasso_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso_model.predict(X_test)

# Evaluate the model using R-squared score
r2 = r2_score(y_test, y_pred)
print(f"R-squared score: {r2:.4f}")

# Display the coefficients of the features
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': lasso_model.coef_})
print('------------Feature Coefficients-------------------')
print(coefficients)
print('------------Lasso Coefficients-------------------------')
print(f'Lasso Coefficients: {lasso_model.coef_}')

print('------------Alpha Values-------------------------')
for alpha in [0.001, 0.1, 0.5, 0.7, 0.8]:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Alpha: {alpha}, R-squared score: {r2:.4f}, Coefficients: {lasso_model.coef_}")