import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # Import necessary for 3D plotting

# Read Excel file
df = pd.read_excel('./main_colors.xlsx', engine='openpyxl')
print("Reading Excel file...")

# Extract color data
colors = df[['L', 'a', 'b']].values  # 'L', 'a', 'b' are the column names in your Excel file
print("Extracting LAB color values...")

# Standardize data
scaler = StandardScaler()
colors = scaler.fit_transform(colors)
print("Standardizing data...")

# Compute silhouette scores to find the optimal k value
silhouette_scores = []
K = range(2, 11)  # Note that silhouette scores require at least 2 clusters
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    preds = kmeanModel.fit_predict(colors)
    silhouette_scores.append(silhouette_score(colors, preds))

# Find the k value with the highest silhouette score
k_best = K[np.argmax(silhouette_scores)]
print("Best number of clusters: ", k_best)
print("Computing silhouette scores for different K values...")

# Plot silhouette scores
plt.plot(K, silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()
print("Plotting silhouette score curve...")

# Perform K-Means clustering with the optimal k value
kmeans = KMeans(n_clusters=k_best)
labels = kmeans.fit_predict(colors)
print("Performing K-Means clustering...")

# Add cluster results to the original data frame
df['Cluster'] = labels
print("Adding cluster results to the original data frame...")

# Save the complete table with clustering results
df.to_excel('./main_colors_clustered.xlsx', index=False)
print("Saved the complete table...")

# Plot scatter plot of clustering results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['L'], df['a'], df['b'], c=labels, cmap='viridis')
ax.set_xlabel('L')
ax.set_ylabel('a')
ax.set_zlabel('b')
ax.set_title('Color Clusters')
plt.show()
print("Plotting scatter plot of clustering results...")
