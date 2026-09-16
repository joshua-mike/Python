import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Helpers'))
from data_dummy import DataDummy

df = DataDummy().create_unsupervised_data()
# Inspect the first few rows of the dataset
print("---------------Initial dataset-------------------")
print(df.head())

# Normalize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back to a DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=['AnnualIncome', 'SpendingScore'])
print("---------------Scaled dataset-------------------")
print(df_scaled.head())

# Initialize the KMeans algorithm with k clusters (start with 3)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
# Fit the model and assign cluster labels
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_
print(f"--------------------Cluster labels assigned with k={k} -------------------------")
print(df.head())

# Visualize the clusters
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Customers')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the WCSS to visualize the elbow point
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-cluster Sum of Squares (WCSS)')
plt.show()