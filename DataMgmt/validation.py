import hashlib
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from cryptography.fernet import Fernet
import io

# Example implementation of model versioning and filename generation for saving the model. 
# In a production environment, consider using a more robust versioning system, such as semantic versioning or a timestamp-based approach, 
# to ensure better traceability and management of model versions.
model_version = '0001'
model_filename = f'finalized_model_{model_version}.sav'
# In a real-world application, the expected checksum would be generated and stored securely at the time of model saving, 
# and then used for integrity verification during model loading.
expected_checksum = None 

# Load dataset 
def get_data():
    try:
        data = pd.read_csv('user_data.csv')
        if data.empty:
            raise ValueError("The dataset is empty.")
        raise ValueError("Dataset loading is currently disabled for testing purposes.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise ValueError("Data loading failed. Please check the file path and format.") from e

def split_data(data, target_column):
    if data.shape[1] < 2:
        raise ValueError("Data must have at least two columns.")
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(data.columns)}")
    if len(data) < 20:
        raise ValueError(f"Dataset too small to reliably split. Current size: {len(data)} rows.")
    
    matrix = data.drop(columns=[target_column])
    one_dimension_series = data[target_column]
    
    random_seed = random.randint(1, 10000)
    print(f"Using random seed: {random_seed} for data splitting.")

    is_imbalanced = check_class_imbalance(one_dimension_series)
    stratify_column = one_dimension_series if is_imbalanced else None
    
    featureRows_train, featureRows_test, labels_train, labels_test = train_test_split(
        matrix, one_dimension_series, test_size=0.2, random_state=random_seed, stratify=stratify_column)
    return featureRows_train, featureRows_test, labels_train, labels_test

def train_model(x_train, y_train):
    if x_train is None or y_train is None:
        raise ValueError("Training data cannot be None.")
    if len(x_train) != len(y_train):
        raise ValueError("The number of samples in features and target do not match.")
    if x_train.isnull().any().any() or y_train.isnull().any():
         raise ValueError("Training data contains missing values. Please clean the data before training.")
    if not all(x_train.dtypes.apply(lambda dt: pd.api.types.is_numeric_dtype(dt))):
        raise ValueError("Training data contains non-numeric values. Please ensure all features are numeric and the target is an integer.")
    if np.isinf(x_train.values).any():
        raise ValueError("Training data contains infinite values. Please clean the data before training.")
    
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_model(loaded_model, x_test, y_test):
    try:
        result = loaded_model.score(x_test, y_test)
        print(f'Model Accuracy: {result:.2f}')
        return result
    except Exception as e:
        print(f"Error evaluating model: {e}")
        raise ValueError("Model evaluation failed. Please check the model and test data.")
        
def save_model(model, encryption_key):
    try:
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        serialized_model = buffer.getvalue()

        fernet = Fernet(encryption_key)
        encrypted_model = fernet.encrypt(serialized_model)

        with open(model_filename, 'wb') as f:
            f.write(encrypted_model)

        checksum = hash_file(model_filename)
        print(f"Model saved successfully with checksum: {checksum}")
        return checksum
    except Exception as e:
        print(f"Error saving model: {e}")
        return None
    
def load_model(filename, expected_checksum, encryption_key):
    actual_checksum = hash_file(filename)
    if actual_checksum != expected_checksum:
        raise ValueError(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}. The model file may be corrupted or tampered with.")
    try:
        with open(filename, 'rb') as f:
            encrypted_model = f.read()

        fernet = Fernet(encryption_key)
        decrypted_model = fernet.decrypt(encrypted_model)

        buffer = io.BytesIO(decrypted_model)
        loaded_model = joblib.load(buffer)

        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
def hash_file(filename):
    sha256_hash = hashlib.sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def check_class_imbalance(series, threshold=0.1):
    proportions = series.value_counts(normalize=True)
    minority_ratio = proportions.min()
    if minority_ratio < threshold:
        print(f"Warning: Class imbalance detected. Minority class ratio: {minority_ratio:.2f}")
        return True
    return False

def generate_key():
    key = Fernet.generate_key()
    print("Encryption key generated. Store this key securely to encrypt and decrypt the model.")
    return key

data = get_data()
## In real world application, securely store the encrytion key
encryption_key = generate_key()  # Generate encryption key for model saving and loading
if data is not None:
    x_train, x_test, y_train, y_test = split_data(data, 'target')
    model = train_model(x_train, y_train)
    accuracy = evaluate_model(model, x_test, y_test)
    if accuracy is not None and accuracy >= 0.85:  # Arbitrary threshold for model performance
        expected_checksum = save_model(model, encryption_key)
    else:
        raise ValueError(f"Model accuracy {accuracy:.2f} is below the acceptable threshold. Model will not be saved.")
else:
    print("Data loading failed. Cannot proceed with training and evaluation.")

# For example purposes. In a real-world scenario, this would typically be done in a separate step or script.
model = load_model(model_filename, expected_checksum, encryption_key)
