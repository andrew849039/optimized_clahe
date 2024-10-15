import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.ticker as ticker
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError

# 设置Seaborn样式
sns.set(style="whitegrid")

# 设置GPU训练
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 设置目录路径
base_dir = "D:/BaiduSyncdisk/work_2024/20240817-colordesign/python/newenv/HRnet"
image_folder = os.path.join(base_dir, "input-P")
output_folder = os.path.join(base_dir, "output-P")
analysis_folder = os.path.join(base_dir, "analysis-P")

# 确保所有目录存在
os.makedirs(output_folder, exist_ok=True)
os.makedirs(analysis_folder, exist_ok=True)

# Step 1: 数据准备
def load_images(image_folder):
    images = []
    for img_name in tqdm(os.listdir(image_folder), desc="Loading images"):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Failed to load image {img_name}")
    return images

def preprocess_images(images):
    processed_images = []
    for img in tqdm(images, desc="Preprocessing images"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # 转换为YUV色彩空间
        img = cv2.resize(img, (128, 128))  # 缩放图像到128x128
        processed_images.append(img)
    return processed_images

# 加载和预处理图像
images = load_images(image_folder)
processed_images = preprocess_images(images)

# 特征提取
def extract_features(image):
    y_channel = cv2.split(image)[0]  # 提取Y通道（亮度）
    l_hist = cv2.calcHist([y_channel], [0], None, [256], [0, 256])
    l_hist = l_hist / l_hist.sum()  # 归一化
    
    contrast = y_channel.std() / y_channel.mean()  # 对比度
    color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])  # 颜色直方图
    color_hist = color_hist.flatten() / color_hist.sum()  # 归一化
    
    gabor_features = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        for sigma in (1, 3):
            kernel = cv2.getGaborKernel((5, 5), sigma, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            fimg = cv2.filter2D(y_channel, cv2.CV_8UC3, kernel)
            gabor_features.append(fimg.mean())
            gabor_features.append(fimg.var())
    
    return np.concatenate([l_hist.flatten(), [contrast], color_hist, gabor_features])

# 特征提取
X = np.array([extract_features(img) for img in processed_images])

# 生成CLAHE增强后的图像（使用默认参数）
def apply_clahe(img, clip_limit=10.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_channel, u, v = cv2.split(img)
    y_channel = clahe.apply(y_channel)
    img_clahe = cv2.merge((y_channel, u, v))
    return cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

# Step 2: 加载预训练模型并应用预测的CLAHE参数
model_path = os.path.join(analysis_folder, 'clahe_param_model_final.h5')
loaded_model = tf.keras.models.load_model(model_path, custom_objects={
    'mse': MeanSquaredError(),
    'mae': MeanAbsoluteError(),
    'rmse': RootMeanSquaredError()
})

y_pred = loaded_model.predict(X)

# 后处理：将预测值限制在有效范围内
y_pred[:, 0] = np.clip(y_pred[:, 0], 1.0, 2.5)  # Clip Limit范围
y_pred[:, 1] = np.clip(y_pred[:, 1], 4, 12)  # Tile Grid Size范围

# 应用CLAHE增强并保存结果
for idx, img in enumerate(processed_images):
    clahe_img = apply_clahe(img)  # 使用默认CLAHE参数
    model_clahe_img = apply_clahe(img, clip_limit=y_pred[idx, 0], tile_grid_size=(int(y_pred[idx, 1]), int(y_pred[idx, 1])))  # 使用模型预测的CLAHE参数
    
    # 保存图像
    cv2.imwrite(os.path.join(output_folder, f"original_clahe_{idx}.png"), clahe_img)
    cv2.imwrite(os.path.join(output_folder, f"model_clahe_{idx}.png"), model_clahe_img)

# Step 3: 可视化对比

# 定义计算颜色复杂度的函数
def color_complexity(img):
    color_hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_hist = color_hist.flatten()
    return np.std(color_hist)  # 颜色复杂度可以通过颜色直方图的标准差来表示

# 定义计算边缘保留的函数
def edge_preservation(original, enhanced):
    # 调整增强后的图像大小以匹配原始图像
    enhanced_resized = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    original_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 100, 200)
    enhanced_edges = cv2.Canny(cv2.cvtColor(enhanced_resized, cv2.COLOR_BGR2GRAY), 100, 200)
    return np.mean(original_edges == enhanced_edges)

# 定义计算SSIM的函数
def calc_ssim(original, processed):
    min_dim = min(original.shape[:2])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    processed_resized = cv2.resize(processed, (original.shape[1], original.shape[0]))
    return ssim(original, processed_resized, multichannel=True, win_size=win_size, channel_axis=-1)

# 更新的calculate_metrics函数
def calculate_metrics(original, enhanced):
    ssim_value = calc_ssim(original, enhanced)
    edge_retention = edge_preservation(original, enhanced)
    color_complex = color_complexity(enhanced)
    
    return ssim_value, edge_retention, color_complex

def plot_comparison(idx, original, clahe_img, model_clahe_img):
    ssim_clahe, edge_clahe, color_clahe = calculate_metrics(original, clahe_img)
    ssim_model, edge_model, color_model = calculate_metrics(original, model_clahe_img)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 设置子图的大小，使整体看起来更加整齐

    # 调整原图长宽比并添加网格
    axes[0, 0].imshow(cv2.cvtColor(cv2.resize(original, (clahe_img.shape[1], clahe_img.shape[0])), cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Original', fontsize=14)
    axes[0, 0].grid(True, linestyle="--", color='gray')
    axes[0, 0].axis('off')
    
    # 显示CLAHE Default图像并添加网格
    axes[0, 1].imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'CLAHE Default', fontsize=14)
    axes[0, 1].grid(True, linestyle="--", color='gray')
    axes[0, 1].axis('off')
    axes[0, 1].set_xlabel(f'SSIM: {ssim_clahe:.2f}\nEdge Retention: {edge_clahe:.2f}\nColor Complexity: {color_clahe:.2f}', fontsize=10)
    
    # 显示CLAHE Predicted图像并添加网格
    axes[0, 2].imshow(cv2.cvtColor(model_clahe_img, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'CLAHE Predicted', fontsize=14)
    axes[0, 2].grid(True, linestyle="--", color='gray')
    axes[0, 2].axis('off')
    axes[0, 2].set_xlabel(f'SSIM: {ssim_model:.2f}\nEdge Retention: {edge_model:.2f}\nColor Complexity: {color_model:.2f}', fontsize=10)
    
    # 绘制SSIM对比
    sns.barplot(ax=axes[1, 0], x=['CLAHE Default', 'CLAHE Predicted'], y=[ssim_clahe, ssim_model], palette="deep", edgecolor=".2")
    axes[1, 0].set_title('SSIM Comparison', fontsize=14)
    for i, v in enumerate([ssim_clahe, ssim_model]):
        axes[1, 0].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    axes[1, 0].set_ylim(0, max(ssim_clahe, ssim_model) * 1.1)

    # 绘制Edge Retention对比
    sns.barplot(ax=axes[1, 1], x=['CLAHE Default', 'CLAHE Predicted'], y=[edge_clahe, edge_model], palette="deep", edgecolor=".2")
    axes[1, 1].set_title('Edge Retention Comparison', fontsize=14)
    for i, v in enumerate([edge_clahe, edge_model]):
        axes[1, 1].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    axes[1, 1].set_ylim(0, max(edge_clahe, edge_model) * 1.1)

    # 绘制Color Complexity对比
    sns.barplot(ax=axes[1, 2], x=['CLAHE Default', 'CLAHE Predicted'], y=[color_clahe, color_model], palette="deep", edgecolor=".2")
    axes[1, 2].set_title('Color Complexity Comparison', fontsize=14)
    for i, v in enumerate([color_clahe, color_model]):
        axes[1, 2].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    axes[1, 2].set_ylim(0, max(color_clahe, color_model) * 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(analysis_folder, f'comparison_{idx}_with_metrics.png'))
    plt.show()

# 可视化前5张图片
for idx in range(min(5, len(images))):
    plot_comparison(
        idx,
        images[idx],  # 使用原始图像而非预处理后的图像
        cv2.imread(os.path.join(output_folder, f"original_clahe_{idx}.png")),
        cv2.imread(os.path.join(output_folder, f"model_clahe_{idx}.png"))
    )

print("Comparison and evaluation complete.")