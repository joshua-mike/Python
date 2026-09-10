import pandas as pd
import numpy as np

class DataDummy:
    def __init__(self):
        self.data = None
        self.k_means_data = None

    def create_dummy_data(self):
        # Create a dummy dataset with 100 rows and 4 columns, including some missing values and outliers
        np.random.seed(0)
        self.data = pd.DataFrame({
            'Feature1': np.random.normal(100, 10, 100).tolist() + [np.nan, 200],  # Normally distributed with an outlier
            'Feature2': np.random.randint(0, 100, 102).tolist(),  # Random integers
            'Category': ['A', 'B', 'C', 'D'] * 25 + [np.nan, 'A'],  # Categorical with some missing values
            'Target': np.random.choice([0, 1], 102).tolist()  # Binary target variable
        })
        print("Dummy data created successfully.")
        return self.data

    def create_k_means_data(self):
        # Create a sample dataset with customer annual income and spending score
        self.k_means_data = {'AnnualIncome': [
            15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 
            20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 
            25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 
            30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5, 
            35,   # Normal points
            80, 85, 90  # Outliers
        ],
        'SpendingScore': [
            39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
            69, 72, 75, 78, 81, 84, 87, 90, 93, 96,
            6, 9, 12, 15, 18, 21, 24, 27, 30, 33,
            5, 8, 11, 14, 17, 20, 23, 26, 29, 32,
            56,   # Normal points
            2, 3, 100  # Outliers
        ]}
        return pd.DataFrame(self.k_means_data)