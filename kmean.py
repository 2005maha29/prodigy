import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Data
# Replace 'customer_data.csv' with the path to your actual data file
data = pd.read_csv(r"C:\Users\paidi\Downloads\customer.csv")

# Display the first few rows of the data
print("Data Overview:")
print(data.head())

# Step 2: Select Relevant Features for Clustering
# Modify the features selected based on your dataset
features = data[['PurchaseAmount', 'NumberOfTransactions']]

# Step 3: Data Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Determine the Optimal Number of Clusters (Elbow Method)
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Step 5: Apply K-means Clustering
# Choose the optimal number of clusters from the elbow plot (e.g., 4 clusters)
optimal_clusters = 4  # Adjust this based on the elbow plot result
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(scaled_features)

# Assign cluster labels to each customer
data['Cluster'] = kmeans.labels_

# Step 6: Analyze and Visualize the Clusters
centroids = kmeans.cluster_centers_

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=data['Cluster'], palette='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('Scaled Purchase Amount')
plt.ylabel('Scaled Number of Transactions')
plt.title('Customer Clusters')
plt.legend()
plt.show()

# Step 7: Save the Results (Optional)
# Save the clustered data to a new CSV file
output_filename = 'clustered_customers.csv'
data.to_csv(output_filename, index=False)
print(f"Clustered data saved to {output_filename}")
