import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# 参数设定
T_COLOR = 50  # 颜色偏移阈值
T_LIGHT = 50  # 低光照阈值
T_BLUR = 10   # 模糊阈值


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


# 图像退化分析函数
def analyze_image(image_path):
    img = read_image_with_pillow(image_path)  # 使用 Pillow 读取图片
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, None, None

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

    return color_cast, low_light, blur


# 图像分类函数
def classify_images(image_folder, output_excel):
    results = []

    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        # 分析图像退化类型
        color_cast, low_light, blur = analyze_image(file_path)
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
