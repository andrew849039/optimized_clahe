import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from math import sqrt, log

# Read Excel file
df = pd.read_excel('./color_clusters.xlsx', engine='openpyxl')

# Extract color data
colors = df[['L', 'a', 'b']].values  # 'L', 'a', 'b' are the column names in your Excel file

# Standardize data
scaler = StandardScaler()
colors = scaler.fit_transform(colors)

# Define maximum number of clusters
max_clusters = 10

# Calculate inertia of the original data
distortions = np.zeros(max_clusters)
for k in range(1, max_clusters + 1):
    kmeanModel = KMeans(n_clusters=k).fit(colors)
    distortions[k - 1] = kmeanModel.inertia_

# Generate B reference datasets and calculate inertia for each
B = 10
reference_distortions = np.zeros((B, max_clusters))
for i in range(B):
    # Generate random reference data
    reference_data = np.random.random_sample(size=colors.shape)

    for k in range(1, max_clusters + 1):
        kmeanModel = KMeans(n_clusters=k).fit(reference_data)
        reference_distortions[i, k - 1] = kmeanModel.inertia_

# Calculate Gap Statistic
gaps = np.mean(np.log(reference_distortions) - np.log(distortions), axis=0)
gaps_plus = gaps + np.std(np.log(reference_distortions), axis=0) * sqrt(1 + 1 / B)
gaps_minus = gaps - np.std(np.log(reference_distortions), axis=0) * sqrt(1 + 1 / B)

# Find the optimal number of clusters
optimal_clusters = np.where(gaps[:-1] >= gaps_plus[1:])[0][0] + 1
print("Optimal clusters: ", optimal_clusters)

# Plot Gap Statistic
plt.figure(figsize=(16, 8))
plt.plot(range(1, max_clusters + 1), gaps, 'bx-')
plt.fill_between(range(1, max_clusters + 1), gaps_plus, gaps_minus, color='gray', alpha=0.2)
plt.xlabel('k')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic vs. k')
plt.show()
