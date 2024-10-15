import cv2
import os
import hashlib
import pandas as pd
import numpy as np
import shutil


def calculate_hash(image_path):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        print(f"无法读取图像：{image_path}")
        return None

    resized_image = cv2.resize(image, (8, 8))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    average_hash = sum([2 ** i for (i, v) in enumerate(threshold_image.flatten()) if v > threshold_image.mean()])
    return average_hash


def find_duplicates(folder_path):
    image_hashes = {}
    duplicate_images = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_hash = calculate_hash(image_path)

                if image_hash is not None:
                    if image_hash in image_hashes:
                        duplicate_images.append((image_path, image_hashes[image_hash]))
                    else:
                        image_hashes[image_hash] = image_path

    return duplicate_images


def remove_duplicate_images(duplicates):
    for image1, image2 in duplicates:
        print(f"重复图像：{image2}")

    choice = input("是否删除这些重复图像？(Y/N): ")
    if choice.lower() == "y":
        for image1, image2 in duplicates:
            print(f"删除图像：{image2}")
            os.remove(image2)
        print("重复图像已删除。")
    else:
        print("不删除重复图像。")


# 设置图像文件夹路径
folder_path = r'D:\Users\LCY\PycharmProjects\pythonProject1\QL_ZH_01'

# 查找重复图像
duplicates = find_duplicates(folder_path)

# 显示重复图像并删除
if len(duplicates) > 0:
    print(f"共找到 {len(duplicates)} 对重复图像：")
    for idx, (image1, image2) in enumerate(duplicates):
        print(f"正在处理第 {idx + 1} 对重复图像...")
        print(f"图像1: {image1}")
        print(f"图像2: {image2}")
        print("-------------------")

    remove_duplicate_images(duplicates)
else:
    print("未找到重复图像。")

# 生成结果表格
result_data = []
for image1, image2 in duplicates:
    result_data.append([image1, image2])

df = pd.DataFrame(result_data, columns=["图像1", "图像2"])
df.to_excel("重复图像结果.xlsx", index=False, encoding='utf-8-sig')

print("重复图像结果已保存到 '重复图像结果.xlsx' 文件中。")
