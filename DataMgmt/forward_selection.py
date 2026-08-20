import  pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']

def forward_selection(X, y):
    remaining_features = set(X.columns)
    selected_features = []
    current_score = 0.0
    best_score = 0.0

    while remaining_features:
        scores_with_candidates = []

        # Loop through remaining features to evaluate their contribution
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            X_train, X_test, y_train, y_test = train_test_split(X[features_to_test], y, test_size=0.2, random_state=42)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict and calculate R^2 score
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            print(f"Evaluating feature: {feature}, R^2 score: {score}")

            # Record the score with the current feature
            scores_with_candidates.append((score, feature))

        # Sort candidates by score in descending order
        scores_with_candidates.sort(reverse=True)
        best_score, best_feature = scores_with_candidates[0]

        # If adding the feature improves the score, add it to the selected features
        if current_score < best_score:
            remaining_features.remove(best_feature)
            selected_features.append(best_feature)
            current_score = best_score
            print(f"Current score improved to {current_score} by adding feature: {best_feature}")
        else:
            break  # No improvement, exit the loop

    return selected_features

# Perform forward selection
best_features = forward_selection(X, y)
print(f"Selected features using forward selection: {best_features}")