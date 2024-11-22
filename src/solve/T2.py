# T2: 退化特性分析
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# 光传输率模型
def transmission_model(d, beta):
    return np.exp(-beta * d)


# 图像退化特性分析
def analyze_degradation(image_path):
    """
    分析单张图像的退化特性，包括颜色偏移、亮度和模糊。
    :param image_path: 图像路径
    :return: 颜色偏移系数、亮度分布、模糊特性
    """
    try:
        img = Image.open(image_path)
        img_cv = np.array(img)
        if img.mode == "RGB":
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None, None, None

    # 提取颜色通道
    b, g, r = cv2.split(img_cv)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 颜色偏移分析
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    beta_r = np.abs(mean_r - mean_g)
    beta_g = np.abs(mean_g - mean_b)
    beta_b = np.abs(mean_b - mean_r)

    # 亮度分析
    brightness = np.mean(gray)

    # 模糊分析（Laplacian 方差）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_level = laplacian.var()

    return (beta_r, beta_g, beta_b), brightness, blur_level


# 参数拟合与退化建模
def fit_degradation_model(image_folder):
    """
    拟合退化模型参数，包括颜色衰减系数、亮度分布和模糊。
    :param image_folder: 图像文件夹路径
    """
    degradation_data = []

    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        # 分析图像退化特性
        color_shift, brightness, blur = analyze_degradation(file_path)
        if color_shift is None:
            continue

        degradation_data.append({
            "image": file_name,
            "color_shift_r": color_shift[0],
            "color_shift_g": color_shift[1],
            "color_shift_b": color_shift[2],
            "brightness": brightness,
            "blur": blur
        })

    return pd.DataFrame(degradation_data)


# 退化特性可视化并保存为图片
def visualize_and_save_degradation(df, output_folder):
    """
    可视化退化特性并将其保存为图片。
    :param df: 图像退化数据 DataFrame
    :param output_folder: 输出图片的文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.figure(figsize=(10, 5))

    # 亮度分布
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df["brightness"], marker='o', label="Brightness")
    plt.title("Brightness Distribution")
    plt.xlabel("Image Index")
    plt.ylabel("Brightness")
    plt.grid()
    plt.legend()

    # 模糊分布
    plt.subplot(1, 2, 2)
    plt.plot(df.index, df["blur"], marker='s', label="Blur Level", color="red")
    plt.title("Blur Level Distribution")
    plt.xlabel("Image Index")
    plt.ylabel("Blur Level")
    plt.grid()
    plt.legend()

    plt.tight_layout()

    # 保存图片到指定文件夹
    output_path = os.path.join(output_folder, "T2_degradation.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"退化特性图已保存至: {output_path}")


# 主函数
if __name__ == "__main__":
    # 更新路径为实际图片文件夹
    image_folder = r"../../data/附件一"
    output_excel = r"../../results/metrics/T2.xlsx"
    output_image_folder = r"../../results/plots"

    # 构建退化模型
    degradation_df = fit_degradation_model(image_folder)

    # 保存数据到 Excel
    degradation_df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"分析结果已保存至: {output_excel}")

    # 可视化退化特性并保存图片
    visualize_and_save_degradation(degradation_df, output_image_folder)

