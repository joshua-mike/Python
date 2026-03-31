import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno
from scipy import stats

# Load dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
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
        numeric_cols = df.select_dtypes(include='number')
        z_scores = np.abs(stats.zscore(numeric_cols))
        df = df[(z_scores < 3).all(axis=1)]
        print("Outliers removed successfully using Z-score.")
    except Exception as e:
        print(f"Error removing outliers: {e}")
    return df

# Cap outliers using quantiles
def cap_outliers(df):
    try:
        upper_limit = df['column_name'].quantile(0.95)
        df['column_name'] = np.where(df['column_name'] > upper_limit, upper_limit, df['column_name'])
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
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        print("Data standardized successfully using Z-score.")
    except Exception as e:
        print(f"Error standardizing data: {e}")
    return df

# One-hot encoding for categorical variables
def one_hot_encoding(df):
     try:
        df = pd.get_dummies(df, columns=['categorical_column_name'])
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

path_to_data = input("Enter the file path to the raw data (e.g., raw_data.csv): ")
df = load_data(path_to_data)
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