import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno
from scipy import stats
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Helpers'))
from data_dummy import DataDummy

# Load dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_dummy_data():
    try:
        dummy_data = DataDummy().create_dummy_data()
        print(f"Dummy data retrieved successfully. {dummy_data.head()}")
        return dummy_data
    except Exception as e:
        print(f"Error generating dummy data: {e}")
        return None

# Visualize missing values
def visualize_missing_values(df):
    msno.matrix(df)
    msno.heatmap(df)

# Drop rows with missing values
def drop_missing_values(df):
     try:
        df = df.dropna()
        print("Rows with missing values dropped successfully.")
     except Exception as e:
        print(f"Error dropping missing values: {e}")
     return df

# Fill missing values with mean where numeric and with "Unknown" where categorical
def fill_missing_values(df):
     try:
        numeric_cols = df.select_dtypes(include='number')
        df[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())
        string_cols = df.select_dtypes(include='str')
        df[string_cols.columns] = string_cols.fillna("Unknown")
        print("Missing values filled successfully.")
     except Exception as e:
        print(f"Error filling missing values: {e}")
     return df

# Identify and remove outliers using Z-score
def remove_outliers_zscore(df):
    try:
        numeric_cols = df.select_dtypes(include='number').to_numpy()
        z_scores = np.abs(stats.zscore(numeric_cols, nan_policy='omit'))
        mask = (z_scores < 3).all(axis=1)
        df = df[mask]
        print(f"Outliers removed successfully using Z-score. Remaining rows: {len(df)}")
    except Exception as e:
        print(f"Error removing outliers: {e}")
    return df

# Cap outliers using quantiles
def cap_outliers(df):
    try:
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            upper_limit = df[col].quantile(0.95)
            lower_limit = df[col].quantile(0.05)
            df[col] = np.clip(df[col], lower_limit, upper_limit)
        print("Outliers capped successfully using quantiles.")
    except Exception as e:
        print(f"Error capping outliers: {e}")
    return df

# Min-Max Scaling
def min_max_scaling(df):
     try:
        numeric_cols = df.select_dtypes(include='number')
        scaler = MinMaxScaler()
        df[numeric_cols.columns] = scaler.fit_transform(numeric_cols)
        print("Data scaled successfully using Min-Max Scaling.")
     except Exception as e:
        print(f"Error scaling data: {e}")
     return df

# Z-score Standardization
def z_score_standardization(df):
    try:
        numeric_cols = df.select_dtypes(include='number')
        scaler = StandardScaler()
        df[numeric_cols.columns] = scaler.fit_transform(numeric_cols)
        print("Data standardized successfully using Z-score.")
    except Exception as e:
        print(f"Error standardizing data: {e}")
    return df

# One-hot encoding for categorical variables
def one_hot_encoding(df):
     try:
        categorical_cols = df.select_dtypes(include='str').columns.tolist()
        df = pd.get_dummies(df, columns=categorical_cols)
        print("One-hot encoding completed successfully.")
     except Exception as e:
        print(f"Error performing one-hot encoding: {e}")
     return df

def save_cleaned_data(df):
    try:
        save_file_path = input("Enter the file path to save the cleaned data (e.g., cleaned_data.csv): ")
        if not os.path.exists(save_file_path):
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        df.to_csv(save_file_path, index=False)
        print(f"Cleaned data saved to {save_file_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")

# path_to_data = input("Enter the file path to the raw data (e.g., raw_data.csv): ")
# df = load_data(path_to_data)
df = get_dummy_data()
if df is None:
    print("Error loading data. Exiting the program.")
    sys.exit(1)
df = drop_missing_values(df)
# Optionally visualize missing values before filling them
# df = visualize_missing_values(df)
df = fill_missing_values(df)
df = remove_outliers_zscore(df)
df = cap_outliers(df)
df = min_max_scaling(df)
df = z_score_standardization(df)
df = one_hot_encoding(df)
save_cleaned_data(df)