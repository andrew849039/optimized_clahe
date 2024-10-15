import os
import sys
import glob
import cv2
import tensorflow as tf
import numpy as np
from absl import logging
from contextlib import contextmanager

# 设置日志级别以隐藏警告
logging.set_verbosity(logging.ERROR)

@contextmanager
def suppress_stdout_stderr():
    """A context manager that suppresses stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# 定义相对路径
base_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(base_dir, "imput")
output_folder = os.path.join(base_dir, "output")
model_dir = os.path.join(base_dir, "HRNet_ade20k-hrnetv2-w48_1")

# 获取所有jpg图像文件
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

# 创建输出目录（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 使用 suppress_stdout_stderr 上下文管理器来抑制警告
with suppress_stdout_stderr():
    model = tf.saved_model.load(model_dir)

# 之后的代码继续执行
for image_path in image_files:
    try:
        image = cv2.imread(image_path)
        height, width = 512, 512
        image_resized = cv2.resize(image, (width, height))
        image_resized = image_resized.astype('float32') / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)
        predictions = model(image_resized)
        if len(predictions.shape) == 4:
            prediction = np.argmax(predictions[0], axis=-1)
        elif len(predictions.shape) == 3:
            prediction = predictions[0]
        else:
            raise ValueError("Unexpected output shape from model: {}".format(predictions.shape))
        sky = (prediction == 3)
        sky_mask = np.repeat(sky[..., np.newaxis], 3, axis=-1)
        image_resized[0, sky_mask] = 0
        final_image_rgb = (image_resized[0] * 255.0).astype(np.uint8)
        final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
        final_image_rgb = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB)
        image_filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, final_image_rgb)
        print(f"Processed image: {image_filename}")
    except cv2.error as e:
        print(f"Error processing image: {image_path}")
        print(e)

print("Processing complete.")