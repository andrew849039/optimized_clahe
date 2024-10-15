import pandas as pd
import numpy as np
from matplotlib.collections import PatchCollection
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

# 1. Load data
df = pd.read_excel('./merged_color_clusters.xlsx')
colors = df[['L', 'a', 'b']]

# 2. Use PCA for dimensionality reduction
pca = PCA(n_components=2)
colors_pca = pca.fit_transform(colors)
colors_pca_df = pd.DataFrame(colors_pca, index=colors.index)

# 3. Use KMeans to find main colors
kmeans = KMeans(n_clusters=3)
kmeans.fit(colors)
centers = kmeans.cluster_centers_

# 4. Save main and support colors
result = pd.DataFrame(columns=['Color Type', 'L', 'a', 'b'])

# Calculate weights for main colors and their matching support colors
for i, center in enumerate(centers):
    main_color = center
    related_colors = colors[kmeans.labels_ == i]
    related_colors_pca_df = pd.DataFrame(pca.transform(related_colors), index=related_colors.index)
    kmeans_support = KMeans(n_clusters=30)
    kmeans_support.fit(related_colors_pca_df)
    support_centers = kmeans_support.cluster_centers_

    for j, support_center in enumerate(support_centers):
        support_distances = ((related_colors_pca_df - support_center) ** 2).sum(axis=1)
        support_color = related_colors.loc[support_distances.idxmin()]

        result = result.append(
            {'Color Type': f'Support Color {j + 1} for Main Color {i + 1}', 'L': support_color[0],
             'a': support_color[1], 'b': support_color[2]}, ignore_index=True)

    result = result.append(
        {'Color Type': f'Main Color {i + 1}', 'L': main_color[0], 'a': main_color[1], 'b': main_color[2]},
        ignore_index=True)

# 6. Save to Excel
for i in range(3):
    group_result_index = result['Color Type'].str.contains(f'Main Color {i + 1}|Support Color.*for Main Color {i + 1}')
    group_result = result.loc[group_result_index, :].copy()
    group_result.to_excel(f'./color_result_group_{i + 1}.xlsx', index=False)

# 7. Visualization
for i in range(3):
    fig, ax = plt.subplots(figsize=(5, 5))

    main_color = result[(result['Color Type'] == f'Main Color {i + 1}')][['L', 'a', 'b']].values[0]
    support_colors = result[(result['Color Type'].str.contains(f'Support Color') & result['Color Type'].str.contains(
        f'for Main Color {i + 1}'))][['L', 'a', 'b']].values

    main_color_rgb = lab2rgb([main_color]).squeeze()
    ax.imshow(np.tile([main_color_rgb], (100, 100, 1)))
    ax.set_title(f'Main Color {i + 1}')
    ax.axis('off')

    support_patches = []
    for j, support_color in enumerate(support_colors):
        support_color_rgb = lab2rgb([support_color]).squeeze()
        patch = plt.Rectangle((j * 10, 0), 10, 10, facecolor=support_color_rgb, edgecolor='none')
        support_patches.append(patch)

    ax.add_collection(PatchCollection(support_patches, match_original=True))

    plt.tight_layout()
    plt.savefig(f'./color_visualization_group_{i + 1}.png')
    plt.close(fig)

print("Completed!")
