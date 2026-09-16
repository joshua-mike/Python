# Density-based spatial clustering of applications with noise (DBSCAN) is a popular clustering algorithm that groups together points that are closely packed together, 
# while marking points that lie alone in low-density regions as outliers.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Helpers'))
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from data_dummy import DataDummy

df = DataDummy().create_unsupervised_data()

# Display the first few rows of the dataset
print(df.head())

# Normalize the dataset using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore'])
print(df_scaled.head())

# Initialize the DBSCAN algorithm with specified parameters
dbscan = DBSCAN(eps=0.5, min_samples=5)
# Fit the model to the scaled data
dbscan.fit(df_scaled)
# Assign cluster labels to the data points
df['Cluster'] = dbscan.labels_
print(df.head())

# Plot the clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='rainbow')
plt.title('DBSCAN Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Tun DBSCAN Parameters: You can experiment with different values of eps and min_samples to see how the clustering results change.
dbscan = DBSCAN(eps=0.7, min_samples=3)
dbscan.fit(df_scaled)
df['Cluster'] = dbscan.labels_

# Plot the new clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='rainbow')
plt.title('DBSCAN Clustering of Customers (Tuned eps=0.7, min_samples=3)')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()



