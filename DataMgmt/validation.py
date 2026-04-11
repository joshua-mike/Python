import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset 
def get_data():
    try:
        data = pd.read_csv('user_data.csv')
        if data.empty:
            raise ValueError("The dataset is empty.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
data = pd.read_csv('user_data.csv')

# Split the dataset into features and target, validate the data and cross-validate the split with a true random seed (Flaw: No checks for data leakage or imbalance in the split)
# Check for data leakage by ensuring that the same random seed is not used across different runs, which could lead to predictable splits and potential overfitting. 
# Additionally, check for class imbalance in the target variable to ensure that the model is trained on a representative sample of the data.
def split_data(data):
    if data is None:
        raise ValueError("Data cannot be empty.")
    if data.shape[1] < 2:
        raise ValueError("Data must have at least two columns.")
    # Verify 20 is a reasonable threshold for the dataset size, as smaller datasets may not provide enough information for training a reliable model. 
    # Consider using techniques like cross-validation or data augmentation if the dataset is too small.
    if len(data) < 20:
        raise ValueError(f"Dataset is too small to reliably split. Current size: {len(data)}. Consider collecting more data.")
    matrix = data.iloc[:, :-1]
    one_demension_series = data.iloc[:, -1]
    random_seed = random.randint(1, 100)
    # Print/log the random seed used for splitting to ensure reproducibility and to allow for debugging if needed.
    print(f"Using random seed: {random_seed} for data splitting.")
    featureRows_train, featureRows_test, labels_train, labels_test = train_test_split(matrix, one_demension_series, test_size = 0.2, random_state = random_seed)
    return featureRows_train, featureRows_test, labels_train, labels_test

# Train a simple logistic regression model (Flaw: No model security checks)
def train_model(X_train, y_train):
    if X_train is None or y_train is None:
        raise ValueError("Training data cannot be empty.")
    model = LogisticRegression()
    model.fit(X_train, y_train)
# Save the model to disk (Flaw: Unencrypted model saving)
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))



# Load the model from disk for later use (Flaw: No integrity checks on the loaded model)
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(f'Model Accuracy: {result:.2f}')