import os
import cv2
import glob
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import time
from absl import logging
import tensorflow as tf
from skimage.color import deltaE_ciede2000
from skimage.feature import canny

# 设置日志级别以隐藏警告
logging.set_verbosity(logging.ERROR)

# 定义相对路径
base_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(base_dir, "input-P")
output_folder = os.path.join(base_dir, "output-Q")
analysis_folder = os.path.join(base_dir, "analysis-Q")
model_dir = os.path.join(base_dir, "HRNet_ade20k-hrnetv2-w48_1")

# 获取所有jpg图像文件
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

# 创建输出目录和分析目录（如果不存在）
os.makedirs(output_folder, exist_ok=True)
os.makedirs(analysis_folder, exist_ok=True)

# 加载HRNet模型
if os.path.exists(model_dir):
    model = tf.saved_model.load(model_dir)
else:
    raise ValueError(f"Model directory {model_dir} does not exist.")

# 初始化数据存储列表
color_fidelity_clahe = []
edge_preservation_clahe = []
ssim_clahe = []
brightness_preservation_clahe = []
contrast_enhancement_clahe = []
artifact_clahe = []

color_fidelity_other = {'hist_eq': [], 'gamma': [], 'retinex': []}
edge_preservation_other = {'hist_eq': [], 'gamma': [], 'retinex': []}
ssim_other = {'hist_eq': [], 'gamma': [], 'retinex': []}
brightness_preservation_other = {'hist_eq': [], 'gamma': [], 'retinex': []}
contrast_enhancement_other = {'hist_eq': [], 'gamma': [], 'retinex': []}
artifact_other = {'hist_eq': [], 'gamma': [], 'retinex': []}

# 定义计算SSIM的函数
def calc_ssim(original, processed):
    min_dim = min(original.shape[:2])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    return ssim(original, processed, multichannel=True, win_size=win_size, channel_axis=-1)

# 其他颜色校正方法
def apply_hist_eq(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def apply_gamma(image, gamma=1.1):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = img_yuv[:, :, 0]
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    y_channel_gamma = cv2.LUT(y_channel, table)
    img_yuv[:, :, 0] = y_channel_gamma
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def apply_retinex(image, sigma=30):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = img_yuv[:, :, 0]
    img_log = np.log1p(y_channel.astype(np.float32))
    img_blur = cv2.GaussianBlur(img_log, (0, 0), sigma)
    retinex = cv2.normalize(np.exp(img_log - img_blur), None, 0, 255, cv2.NORM_MINMAX)
    img_yuv[:, :, 0] = retinex.astype(np.uint8)
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

# 计算颜色保真度
def color_fidelity(original, processed):
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    processed_lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    delta_e = deltaE_ciede2000(original_lab, processed_lab)
    return np.mean(delta_e)

# 计算边缘保留度
def edge_preservation(original, processed):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    original_edges = canny(original_gray)
    processed_edges = canny(processed_gray)
    return ssim(original_edges, processed_edges, multichannel=False)

# 计算亮度保留度
def brightness_preservation(original, processed):
    original_yuv = cv2.cvtColor(original, cv2.COLOR_BGR2YUV)
    processed_yuv = cv2.cvtColor(processed, cv2.COLOR_BGR2YUV)
    original_brightness = np.mean(original_yuv[:, :, 0])
    processed_brightness = np.mean(processed_yuv[:, :, 0])
    return abs(original_brightness - processed_brightness)

# 计算对比度增强指数
def contrast_enhancement(original, processed):
    original_contrast = original.std()
    processed_contrast = processed.std()
    return processed_contrast / original_contrast

# 伪影检测
def artifact_detection(original, processed):
    diff = cv2.absdiff(original, processed)
    return np.mean(diff)

# 图像处理和天空去除
for image_path in image_files:
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}. Skipping this file.")
            continue

        height, width = image.shape[:2]
        if height > 512 or width > 512:
            scale = min(512 / height, 512 / width)
            height = int(height * scale)
            width = int(width * scale)
            image_resized = cv2.resize(image, (width, height))
        else:
            image_resized = image
        image_resized = image_resized.astype('float32') / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)

        if len(image_resized.shape) == 4:
            predictions = model(image_resized)
        else:
            raise ValueError(f"Unexpected input shape: {image_resized.shape}")
        
        if len(predictions.shape) == 4:
            prediction = np.argmax(predictions[0], axis=-1)
        elif len(predictions.shape) == 3:
            prediction = predictions[0]
        else:
            raise ValueError(f"Unexpected output shape from model: {predictions.shape}")
        
        sky_label = 3
        sky = (prediction == sky_label)
        sky_mask = np.repeat(sky[..., np.newaxis], 3, axis=-1)
        image_resized[0, sky_mask] = 0
        final_image_bgr = (image_resized[0] * 255.0).astype(np.uint8)

        # CLAHE处理
        img_yuv = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
        y_channel = img_yuv[:, :, 0]
        y_channel_clahe = clahe.apply(y_channel)
        img_yuv[:, :, 0] = y_channel_clahe
        final_image_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # 记录指标
        color_fidelity_clahe.append(color_fidelity(final_image_bgr, final_image_clahe))
        edge_preservation_clahe.append(edge_preservation(final_image_bgr, final_image_clahe))
        ssim_clahe.append(calc_ssim(final_image_bgr, final_image_clahe))
        brightness_preservation_clahe.append(brightness_preservation(final_image_bgr, final_image_clahe))
        contrast_enhancement_clahe.append(contrast_enhancement(final_image_bgr, final_image_clahe))
        artifact_clahe.append(artifact_detection(final_image_bgr, final_image_clahe))

        # 其他方法处理
        other_methods = {'hist_eq': apply_hist_eq, 'gamma': lambda img: apply_gamma(img), 'retinex': apply_retinex}
        for method_name, method in other_methods.items():
            final_image_other = method(final_image_bgr)

            color_fidelity_other[method_name].append(color_fidelity(final_image_bgr, final_image_other))
            edge_preservation_other[method_name].append(edge_preservation(final_image_bgr, final_image_other))
            ssim_other[method_name].append(calc_ssim(final_image_bgr, final_image_other))
            brightness_preservation_other[method_name].append(brightness_preservation(final_image_bgr, final_image_other))
            contrast_enhancement_other[method_name].append(contrast_enhancement(final_image_bgr, final_image_other))
            artifact_other[method_name].append(artifact_detection(final_image_bgr, final_image_other))

            output_path = os.path.join(output_folder, f"{method_name}_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, final_image_other)

        clahe_output_path = os.path.join(output_folder, f"clahe_{os.path.basename(image_path)}")
        cv2.imwrite(clahe_output_path, final_image_clahe)

        analysis_path = os.path.join(analysis_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_analysis.txt")
        with open(analysis_path, 'w') as f:
            f.write(f"Color Fidelity (Original vs CLAHE): {color_fidelity_clahe[-1]}\n")
            f.write(f"Edge Preservation (CLAHE): {edge_preservation_clahe[-1]}\n")
            f.write(f"SSIM (CLAHE): {ssim_clahe[-1]}\n")
            f.write(f"Brightness Preservation (CLAHE): {brightness_preservation_clahe[-1]}\n")
            f.write(f"Contrast Enhancement (CLAHE): {contrast_enhancement_clahe[-1]}\n")
            f.write(f"Artifact Detection (CLAHE): {artifact_clahe[-1]}\n")
            for method_name in other_methods.keys():
                f.write(f"Color Fidelity (Original vs {method_name}): {color_fidelity_other[method_name][-1]}\n")
                f.write(f"Edge Preservation ({method_name}): {edge_preservation_other[method_name][-1]}\n")
                f.write(f"SSIM ({method_name}): {ssim_other[method_name][-1]}\n")
                f.write(f"Brightness Preservation ({method_name}): {brightness_preservation_other[method_name][-1]}\n")
                f.write(f"Contrast Enhancement ({method_name}): {contrast_enhancement_other[method_name][-1]}\n")
                f.write(f"Artifact Detection ({method_name}): {artifact_other[method_name][-1]}\n")

        print(f"Analysis complete for image: {image_path}")

    except cv2.error as e:
        print(f"Error processing image: {image_path}")
        print(e)

# 计算平均值
average_values_clahe = {
    'Color Fidelity': np.mean(color_fidelity_clahe),
    'Edge Preservation': np.mean(edge_preservation_clahe),
    'SSIM': np.mean(ssim_clahe),
    'Brightness Preservation': np.mean(brightness_preservation_clahe),
    'Contrast Enhancement': np.mean(contrast_enhancement_clahe),
    'Artifact Detection': np.mean(artifact_clahe)
}

average_values_other = {method_name: {
    'Color Fidelity': np.mean(color_fidelity_other[method_name]),
    'Edge Preservation': np.mean(edge_preservation_other[method_name]),
    'SSIM': np.mean(ssim_other[method_name]),
    'Brightness Preservation': np.mean(brightness_preservation_other[method_name]),
    'Contrast Enhancement': np.mean(contrast_enhancement_other[method_name]),
    'Artifact Detection': np.mean(artifact_other[method_name])
} for method_name in other_methods.keys()}

# 可视化结果并显示平均值
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.ravel()

metrics = {
    'Color Fidelity': (color_fidelity_clahe, color_fidelity_other),
    'Edge Preservation': (edge_preservation_clahe, edge_preservation_other),
    'SSIM': (ssim_clahe, ssim_other),
    'Brightness Preservation': (brightness_preservation_clahe, brightness_preservation_other),
    'Contrast Enhancement': (contrast_enhancement_clahe, contrast_enhancement_other),
    'Artifact Detection': (artifact_clahe, artifact_other)
}

for i, (metric_name, (clahe_values, other_values)) in enumerate(metrics.items()):
    axs[i].plot(clahe_values, label=f'CLAHE (Avg: {average_values_clahe[metric_name]:.4f})')
    for method_name in other_methods.keys():
        axs[i].plot(other_values[method_name], label=f'{method_name} (Avg: {average_values_other[method_name][metric_name]:.4f})')
    axs[i].set_title(metric_name)
    axs[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(analysis_folder, 'metrics_comparison.png'))
plt.show()
print("Visualization saved.")
