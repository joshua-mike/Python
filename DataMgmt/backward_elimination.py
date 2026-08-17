import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'Pass': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1] # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']

# Add a constant to the model (intercept)
X = pd.DataFrame(sm.add_constant(X))

# Fit the model using Ordinary Least Squares (OLS) regression
model = sm.OLS(y, X).fit()

# Display the summary, including p-values for each feature
print(model.summary())

# Perform backward elimination
significance_level = 0.05
while True:
    # Fit the model
    model = sm.OLS(y, X).fit()
    # Get the highest p-value in the model
    max_p_value = model.pvalues.max()

    # Check if the highest p-value is greater than the significance level
    if max_p_value > significance_level:
        # Identify the feature with the highest p-value
        feature_to_remove = model.pvalues.idxmax()
        print(f"Removing feature '{feature_to_remove}' with p-value {max_p_value}")

        # Drop the feature from the dataset
        X = X.drop(columns=[feature_to_remove])
    else:
        break

# Display the final model summary after backward elimination
print(model.summary())