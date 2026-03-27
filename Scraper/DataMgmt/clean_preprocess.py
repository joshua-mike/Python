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
     return df.dropna()

# Fill missing values with mean
def fill_missing_values(df):
     return df.fillna(df.mean())

# Identify and remove outliers using Z-score
def remove_outliers_zscore(df):
    z_scores = np.abs(stats.zscore(df))
    df_no_outliers = df[(z_scores < 3).all(axis=1)]
    return df_no_outliers

# Cap outliers using quantiles
def cap_outliers(df):
    upper_limit = df['column_name'].quantile(0.95)
    df['column_name'] = np.where(df['column_name'] > upper_limit, upper_limit, df['column_name'])
    return df

# Min-Max Scaling
def min_max_scaling(df):
     scaler = MinMaxScaler()
     df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
     return df_scaled

# Z-score Standardization
def z_score_standardization(df):
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_standardized

# One-hot encoding for categorical variables
def one_hot_encoding(df):
     return pd.get_dummies(df, columns=['categorical_column_name'])

def save_cleaned_data(df, file_name):
    df.to_csv(file_name, index=False)
    print(f"Cleaned data saved to {file_name}")

df = load_data('raw_data.csv')
df = drop_missing_values(df)
# Optionally visualize missing values before filling them
# df = visualize_missing_values(df)
df = fill_missing_values(df)
df = remove_outliers_zscore(df)
df = cap_outliers(df)
df = min_max_scaling(df)
df =z_score_standardization(df)
df = one_hot_encoding(df)
save_cleaned_data(df, 'cleaned_data.csv')