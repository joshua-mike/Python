import pandas as pd
import numpy as np

class DataDummy:
    def __init__(self):
        self.data = None

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