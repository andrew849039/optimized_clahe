import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D

# 设置GPU训练
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 设置目录路径
base_dir = "./HRnet"
image_folder = os.path.join(base_dir, "input-p")
output_folder = os.path.join(base_dir, "output-P")
analysis_folder = os.path.join(base_dir, "analysis-P")
model_dir = os.path.join(base_dir, "HRNet_ade20k-hrnetv2-w48_1")

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
        processed_images.append(img)
    return processed_images

# 加载和预处理图像
images = load_images(image_folder)
processed_images = preprocess_images(images)

# Step 2: 数据扩充及特征提取
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def extract_features(image):
    y_channel = cv2.split(image)[0]  # 提取Y通道（亮度）
    resized_img = cv2.resize(y_channel, (224, 224))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)  # 将单通道灰度图转换为三通道图像
    
    # 使用VGG16提取高层特征
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg_features = vgg_model.predict(np.expand_dims(resized_img, axis=0))
    vgg_features = GlobalAveragePooling2D()(vgg_features)
    vgg_features = vgg_features.numpy().flatten()  # 将特征向量转换为NumPy数组并展平
    
    l_hist = cv2.calcHist([y_channel], [0], None, [256], [0, 256])
    l_hist = l_hist / l_hist.sum()  # 归一化
    l_hist = l_hist.flatten()  # 展平直方图特征
    
    contrast = y_channel.std() / y_channel.mean()  # 计算图像的对比度
    
    return np.concatenate([vgg_features, l_hist, [contrast]])

def prepare_dataset(images, params, extreme_weight=20):
    X = []
    y_clip_limit = []
    y_tile_grid_size = []
    for img, param in zip(images, params):
        features = extract_features(img)
        X.append(features)
        y_clip_limit.append(param[0])
        y_tile_grid_size.append(param[1])
        if param[0] > 2.0 or param[1] < 5 or param[1] > 10:
            for _ in range(extreme_weight):
                X.append(features)
                y_clip_limit.append(param[0])
                y_tile_grid_size.append(param[1])
    return np.array(X), np.array(y_clip_limit), np.array(y_tile_grid_size)

# Step 3: 构建和训练优化的神经网络模型
def build_optimized_model(input_shape):
    input_layer = tf.keras.layers.Input(shape=(input_shape,))
    
    x = tf.keras.layers.Dense(2048, activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    residual = tf.keras.layers.Dense(512, activation='relu')(input_layer)
    x = tf.keras.layers.add([x, residual]) 
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    clip_limit_output = tf.keras.layers.Dense(1, activation='linear', name='clip_limit')(x)
    tile_grid_size_output = tf.keras.layers.Dense(1, activation='linear', name='tile_grid_size')(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=[clip_limit_output, tile_grid_size_output])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                  loss={'clip_limit': 'mse', 'tile_grid_size': 'mse'},
                  metrics={'clip_limit': tf.keras.metrics.RootMeanSquaredError(),
                           'tile_grid_size': tf.keras.metrics.RootMeanSquaredError()})

    return model

def generate_clahe_params(images):
    params = []
    for img in images:
        avg_brightness = np.mean(cv2.split(img)[0])
        clip_limit = np.random.uniform(1.0, 2.5)
        tile_grid_size = np.random.randint(4, 13)
        params.append([clip_limit, tile_grid_size])
    return np.array(params)

best_clahe_params = generate_clahe_params(processed_images)
X, y_clip_limit, y_tile_grid_size = prepare_dataset(processed_images, best_clahe_params)

X_train, X_val, y_clip_limit_train, y_clip_limit_val, y_tile_grid_size_train, y_tile_grid_size_val = train_test_split(
    X, y_clip_limit, y_tile_grid_size, test_size=0.2, random_state=42)

param_model = build_optimized_model(X_train.shape[1])

early_stopping = EarlyStopping(monitor='val_clip_limit_root_mean_squared_error', mode='min', patience=20, restore_best_weights=True) 
reduce_lr = ReduceLROnPlateau(monitor='val_clip_limit_root_mean_squared_error', mode='min', factor=0.5, patience=8, min_lr=0.00001, verbose=1)

history = param_model.fit(X_train, {'clip_limit': y_clip_limit_train, 'tile_grid_size': y_tile_grid_size_train}, 
                          epochs=200, validation_data=(X_val, {'clip_limit': y_clip_limit_val, 'tile_grid_size': y_tile_grid_size_val}),
                          batch_size=64, callbacks=[early_stopping, reduce_lr])

# 保存模型为H5格式
model_save_path = os.path.join(analysis_folder, "optimized_clahe_param_model.h5")
param_model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Step 4: 可视化训练过程和评估结果

def plot_training_history(history):
    plt.figure(figsize=(18, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history.history['clip_limit_loss'], label='Training Clip Limit Loss')
    plt.plot(history.history['val_clip_limit_loss'], label='Validation Clip Limit Loss') 
    plt.title('Clip Limit Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)  
    plt.plot(history.history['clip_limit_root_mean_squared_error'], label='Training Clip Limit RMSE')
    plt.plot(history.history['val_clip_limit_root_mean_squared_error'], label='Validation Clip Limit RMSE')
    plt.title('Clip Limit RMSE During Training') 
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')  
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history.history['tile_grid_size_loss'], label='Training Tile Grid Size Loss')
    plt.plot(history.history['val_tile_grid_size_loss'], label='Validation Tile Grid Size Loss')
    plt.title('Tile Grid Size Loss During Training') 
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  

    plt.subplot(2, 2, 4)
    plt.plot(history.history['tile_grid_size_root_mean_squared_error'], label='Training Tile Grid Size RMSE') 
    plt.plot(history.history['val_tile_grid_size_root_mean_squared_error'], label='Validation Tile Grid Size RMSE')
    plt.title('Tile Grid Size RMSE During Training')
    plt.xlabel('Epoch') 
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()  
    plt.savefig(os.path.join(analysis_folder, 'optimized_training_history.png'))
    plt.show()

plot_training_history(history)

# 预测并可视化预测结果与真实值的对比
y_pred_clip_limit, y_pred_tile_grid_size = param_model.predict(X_val)

def plot_predicted_vs_actual(y_true_clip_limit, y_pred_clip_limit, y_true_tile_grid_size, y_pred_tile_grid_size):  
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].scatter(y_true_clip_limit, y_pred_clip_limit, alpha=0.5)
    axs[0].plot([1.0, 2.5], [1.0, 2.5], 'r--')  # 添加对角线
    axs[0].set_xlabel('Actual Clip Limit') 
    axs[0].set_ylabel('Predicted Clip Limit')
    axs[0].set_title('Clip Limit: Predicted vs. Actual') 
    axs[0].set_xlim(1.0, 2.5)
    axs[0].set_ylim(1.0, 2.5)  
    
    axs[1].scatter(y_true_tile_grid_size, y_pred_tile_grid_size, alpha=0.5)
    axs[1].plot([4, 12], [4, 12], 'r--')
    axs[1].set_xlabel('Actual Tile Grid Size')  
    axs[1].set_ylabel('Predicted Tile Grid Size')
    axs[1].set_title('Tile Grid Size: Predicted vs. Actual') 
    axs[1].set_xlim(4, 12) 
    axs[1].set_ylim(4, 12)
    
    plt.tight_layout()   
    plt.savefig(os.path.join(analysis_folder, 'predicted_vs_actual.png'))
    plt.show()

plot_predicted_vs_actual(y_clip_limit_val, y_pred_clip_limit, y_tile_grid_size_val, y_pred_tile_grid_size)

# 绘制预测误差直方图  
def plot_error_histograms(y_true_clip_limit, y_pred_clip_limit, y_true_tile_grid_size, y_pred_tile_grid_size):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    clip_limit_error = y_pred_clip_limit - y_true_clip_limit 
    axs[0].hist(clip_limit_error, bins=50)
    axs[0].set_xlabel('Prediction Error') 
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Clip Limit Prediction Error Distribution')  
    
    tile_grid_size_error = y_pred_tile_grid_size - y_true_tile_grid_size
    axs[1].hist(tile_grid_size_error, bins=50) 
    axs[1].set_xlabel('Prediction Error')
    axs[1].set_ylabel('Frequency') 
    axs[1].set_title('Tile Grid Size Prediction Error Distribution')
    
    plt.tight_layout()    
    plt.savefig(os.path.join(analysis_folder, 'prediction_error_histograms.png'))
    plt.show() 

plot_error_histograms(y_clip_limit_val, y_pred_clip_limit, y_tile_grid_size_val, y_pred_tile_grid_size)

print("Training and evaluation complete.")
