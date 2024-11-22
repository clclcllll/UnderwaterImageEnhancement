# 基于 T3 的图像增强方法的合并版T4
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image


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


# PSNR 计算
def calculate_psnr(image, reference):
    mse = np.mean((image.astype(np.float64) - reference.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


# UCIQE 计算
def calculate_uciqe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    chroma = np.sqrt(a.astype(np.float64) ** 2 + b.astype(np.float64) ** 2)
    return np.std(chroma) + np.std(l) + np.mean(l) / (np.mean(chroma) + 1e-6)


# UIQM 计算
def calculate_uiqm(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return 0.5 * np.std(h.astype(np.float64)) + 0.25 * np.std(s.astype(np.float64)) + 0.25 * np.mean(
        v.astype(np.float64))


# 颜色校正模块
def enhance_color_correction(img):
    b, g, r = cv2.split(img)
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    mean_avg = (mean_r + mean_g + mean_b) / 3
    gamma_r, gamma_g, gamma_b = mean_avg / mean_r, mean_avg / mean_g, mean_avg / mean_b
    r = np.clip(r * gamma_r, 0, 255).astype(np.uint8)
    g = np.clip(g * gamma_g, 0, 255).astype(np.uint8)
    b = np.clip(b * gamma_b, 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r))


# 亮度增强模块
def enhance_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv_enhanced = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)


# 去模糊模块
def enhance_sharpness(img):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


# 联合增强模型
def enhance_image(img):
    img_color_corrected = enhance_color_correction(img)
    img_brightness_enhanced = enhance_brightness(img_color_corrected)
    img_sharp = enhance_sharpness(img_brightness_enhanced)
    return img_sharp


# 图像增强与验证
def enhance_and_save(image_folder, output_excel, output_image_folder):
    # 确保输出目录存在
    os.makedirs(output_image_folder, exist_ok=True)

    results = []
    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        if not os.path.isfile(file_path):
            print(f"跳过非文件: {file_name}")
            continue

        # 读取图像
        img = read_image_with_pillow(file_path)
        if img is None:
            print(f"无法处理图像: {file_name}")
            continue

        img = img.astype(np.uint8)
        reference = np.full_like(img, 128, dtype=np.uint8)

        # 原始指标
        psnr_before = calculate_psnr(img, reference)
        uciqe_before = calculate_uciqe(img)
        uiqm_before = calculate_uiqm(img)

        # 增强图像
        enhanced_img = enhance_image(img)

        # 增强后指标
        psnr_after = calculate_psnr(enhanced_img, reference)
        uciqe_after = calculate_uciqe(enhanced_img)
        uiqm_after = calculate_uiqm(enhanced_img)

        # 保存增强后的图像到指定目录
        enhanced_img_name = os.path.join(output_image_folder, f"enhanced_{file_name}")
        try:
            cv2.imwrite(enhanced_img_name, enhanced_img)
            print(f"保存增强图像: {enhanced_img_name}")
        except Exception as e:
            print(f"保存增强图像失败: {file_name}, 错误: {e}")
            continue

        # 保存结果
        results.append({
            "image file name": file_name,
            "Degraded Image Classification": None,
            "PSNR": psnr_before,
            "UCIQE": uciqe_before,
            "UIQM": uiqm_before,
            "PSNR-IM": psnr_after,
            "UCIQE-IM": uciqe_after,
            "UIQM-IM": uiqm_after,
        })

    # 保存到 Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"增强结果已保存至 {output_excel}")


# 主函数
if __name__ == "__main__":
    image_folder = r"../../data/附件一"
    output_excel = r"../../results/metrics/T4_ordinary_basedT3.xlsx"
    output_image_folder = r"../../results/T4enhanced_ordinary_basedT3"  # 指定图片保存路径
    enhance_and_save(image_folder, output_excel, output_image_folder)
