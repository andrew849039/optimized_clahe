import os
from skimage.color import rgb2lab
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_image_pixels(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((150, 150))
    img_np = np.array(img)
    pixels = img_np.reshape(-1, 3)
    pixels = pixels[np.mean(pixels, axis=1) > 30]
    pixels = rgb2lab(pixels / 255.0)
    return pixels

def get_cluster_colors(image_path, max_clusters=30):
    pixels = get_image_pixels(image_path)
    if len(pixels) == 0:
        return None, None

    kmeans = KMeans(n_clusters=min(max_clusters, len(pixels)))
    kmeans.fit(pixels)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    color_ratios = counts / np.sum(counts)

    return cluster_centers, color_ratios

def process_image(image_path, output_folder, max_clusters=30):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    cluster_centers, color_ratios = get_cluster_colors(image_path, max_clusters)
    if cluster_centers is not None and color_ratios is not None:
        data = []
        for i, color in enumerate(cluster_centers):
            ratio = color_ratios[i]
            data.append([image_name, color[0], color[1], color[2], ratio])

        save_colors_to_excel(data, output_folder, image_name)
        print(f"Processed image: {image_name}")
    else:
        print(f"Error processing image: {image_name}")

def save_colors_to_excel(data, output_folder, image_name):
    df = pd.DataFrame(data, columns=['Image', 'L', 'a', 'b', 'Ratio'])
    excel_path = os.path.join(output_folder, f"{image_name}_color_clusters.xlsx")
    df.to_excel(excel_path, index=False)

if __name__ == '__main__':
    image_folder = r"D:\Users\LCY\PycharmProjects\pythonProject1\QL_ZH_01_HX"
    output_folder = r"D:\Users\LCY\PycharmProjects\pythonProject1\QL_ZH_03"
    os.makedirs(output_folder, exist_ok=True)
    max_clusters = 30

    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder)
                   if file.endswith(".jpg") or file.endswith(".png")]

    with ThreadPoolExecutor() as executor:
        futures = []
        for image_path in image_files:
            future = executor.submit(process_image, image_path, output_folder, max_clusters)
            futures.append(future)

        for future in as_completed(futures):
            future.result()

    print("Processing complete.")
