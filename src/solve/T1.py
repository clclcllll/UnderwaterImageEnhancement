import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# 参数设定
T_COLOR = 50  # 颜色偏移阈值
T_LIGHT = 50  # 低光照阈值
T_BLUR = 10  # 模糊阈值


# 使用 Pillow 读取图片并转换为 OpenCV 格式
def read_image_with_pillow(image_path):
    try:
        img = Image.open(image_path)
        img_cv = np.array(img)
        if img.mode == "RGB":
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        return img_cv
    except Exception as e:
        print(f"无法读取图片: {image_path}, 错误: {e}")
        return None


# 计算 PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(image, reference):
    image = image.astype(np.float64)  # 确保数据类型为浮点数
    reference = reference.astype(np.float64)
    mse = np.mean((image - reference) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


# 计算 UCIQE 指标
def calculate_uciqe(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(image)
    chroma = np.sqrt(a_channel ** 2 + b_channel ** 2).astype(np.float64)  # 转换数据类型
    ucq = np.mean(l_channel.astype(np.float64)) * np.std(chroma) / (np.mean(chroma) + 1e-6)
    return ucq


# 计算 UIQM (Underwater Image Quality Metric)
def calculate_uiqm(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    uiqm = 0.5 * np.std(h.astype(np.float64)) + 0.25 * np.std(s.astype(np.float64)) + 0.25 * np.mean(
        v.astype(np.float64))
    return uiqm


# 图像退化分析函数
def analyze_image(image_path):
    img = read_image_with_pillow(image_path)  # 使用 Pillow 读取图片
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, None, None, None, None

    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 颜色偏移分析
    mean_b, mean_g, mean_r = cv2.mean(img)[:3]
    rgb_diff = max(abs(mean_r - mean_g), abs(mean_g - mean_b), abs(mean_b - mean_r))
    color_cast = rgb_diff > T_COLOR

    # 低光照分析
    mean_light = np.mean(gray.astype(np.float64))  # 确保使用浮点数计算
    low_light = mean_light < T_LIGHT

    # 模糊分析
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    blur = variance < T_BLUR

    # 计算 PSNR
    reference = np.full_like(img, 128)  # 假设为全灰的图像
    psnr = calculate_psnr(img, reference)

    # 计算 UCIQE 和 UIQM
    uciqe = calculate_uciqe(img)
    uiqm = calculate_uiqm(img)

    return color_cast, low_light, blur, psnr, uciqe, uiqm


# 图像分类函数
def classify_images(image_folder, output_excel):
    results = []

    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        # 分析图像退化类型和指标
        color_cast, low_light, blur, psnr, uciqe, uiqm = analyze_image(file_path)
        if color_cast is None:
            continue

        # 分类
        if color_cast:
            classification = "Color Cast"
        elif low_light:
            classification = "Low Light"
        elif blur:
            classification = "Blur"
        else:
            classification = "Clear"

        # 记录结果
        results.append({
            "image file name": file_name,
            "Degraded Image Classification": classification,
            "PSNR": None,
            "UCIQE": None,
            "UIQM": None
        })

    # 保存到 Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"分类结果已保存至 {output_excel}")


# 主函数
if __name__ == "__main__":
    image_folder = r"../../data/附件一"
    output_excel = r"../../results/metrics/T1.xlsx"
    classify_images(image_folder, output_excel)
