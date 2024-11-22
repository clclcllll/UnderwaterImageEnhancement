#  T4_optimizing_basedT3.py
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# 计算 PSNR（峰值信噪比）
def calculate_psnr(image, reference):
    mse = np.mean((image.astype(np.float64) - reference.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


# 计算 UCIQE 指标
def calculate_uciqe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    chroma = np.sqrt(a_channel.astype(np.float64) ** 2 + b_channel.astype(np.float64) ** 2)
    ucq = np.mean(l_channel.astype(np.float64)) * np.std(chroma) / (np.mean(chroma) + 1e-6)
    return ucq


# 计算 UIQM 指标
def calculate_uiqm(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    uiqm = 0.5 * np.std(h.astype(np.float64)) + 0.25 * np.std(s.astype(np.float64)) + 0.25 * np.mean(
        v.astype(np.float64))
    return uiqm


# 从 enhance.py 中引入的图像增强函数
def enhance_underwater_image(img):
    """
    水下图像增强，包括白平衡调整、去雾、直方图均衡化和伽马校正。
    参数:
        img (numpy.ndarray): 输入图像，BGR 格式。
    返回:
        numpy.ndarray: 增强后的图像。
    """
    # 按照 enhance.py 中定义的增强流水线
    img = super_resolution(img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_image)
    img = sharpen_with_kernel(img)
    img = dehaze(img)
    if mean_brightness < 50:
        img = gamma_correction(img, gamma=1.8)
    img = white_balance(img)
    img = histogram_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8))
    return img


# 伽马校正
def gamma_correction(img, gamma, clip_limit=0.05):
    """
    对图像进行伽马校正，并限制亮度范围。
    """
    # 将图像转换为浮点数并归一化到 [0, 1]
    img = img.astype(np.float32) / 255.0

    # 自动调整伽马值
    mean_brightness = np.mean(img)
    if mean_brightness > 0.6:  # 过亮时增加伽马值
        gamma += 0.5

    # 伽马校正
    img = np.power(img, 1 / gamma)

    # 限制最大值
    img = np.clip(img, 0, 1)

    return (img * 255).astype(np.uint8)


def super_resolution(img):
    """
    使用预训练的超分辨率模型提高分辨率。
    参数:
        img (numpy.ndarray): 输入低分辨率图像。
    返回:
        numpy.ndarray: 高分辨率图像。
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    # 加载预训练的 ESPCN 模型
    model_path = "ESPCN_x2.pb"  # 替换为实际模型路径
    try:
        sr.readModel(model_path)
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")
    sr.setModel("espcn", 2)  # 使用 ESPCN 模型，放大倍数为 2
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("输入图像必须是灰度图像或 3 通道彩色图像。")
    height, width = img.shape[:2]
    new_height = (height // 2) * 2
    new_width = (width // 2) * 2
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    try:
        high_res_img = sr.upsample(resized_img)
    except Exception as e:
        raise RuntimeError(f"超分辨率处理失败: {e}")
    high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_RGB2BGR)
    return high_res_img


def sharpen_with_kernel(img):
    """
    使用卷积核锐化图像。
    参数:
        img (numpy.ndarray): 输入图像，BGR 格式。
    返回:
        numpy.ndarray: 锐化后的图像。
    """
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img


def white_balance(img):
    """
    使用改进的白平衡算法调整图像颜色，避免偏色或过度调整，特别适配水下图像。
    参数:
        img (numpy.ndarray): 输入图像，BGR 格式。
    返回:
        numpy.ndarray: 调整后的图像。
    """
    img = img.astype(np.float32) / 255.0
    mean_r = np.mean(img[:, :, 2][img[:, :, 2] > 0.01])
    mean_g = np.mean(img[:, :, 1][img[:, :, 1] > 0.01])
    mean_b = np.mean(img[:, :, 0][img[:, :, 0] > 0.01])
    mean_gray = (mean_r + mean_g + mean_b) / 3
    correction_r = min(mean_gray / (mean_r + 1e-6), 1.5)
    correction_g = mean_gray / (mean_g + 1e-6)
    correction_b = mean_gray / (mean_b + 1e-6)
    if mean_g > mean_r and mean_b > mean_r:
        correction_r *= 0.85
    img[:, :, 2] *= correction_r
    img[:, :, 1] *= correction_g
    img[:, :, 0] *= correction_b
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def dehaze(img):
    """
    改进的去雾方法，适配水下图像。
    参数:
        img (numpy.ndarray): 输入图像，BGR 格式。
    返回:
        numpy.ndarray: 去雾后的图像。
    """
    img = img.astype(np.float32) / 255.0
    dark_channel = get_dark_channel(img, window_size=15)
    atmospheric_light = estimate_atmospheric_light(img, dark_channel)
    omega = 0.95
    transmission = estimate_transmission(img, atmospheric_light, omega)
    transmission = refine_transmission(transmission, img)
    result = recover_scene_radiance(img, atmospheric_light, transmission)
    return (result * 255).astype(np.uint8)


# 暗通道先验计算
def get_dark_channel(img, window_size):
    """
    计算暗通道图像。
    Args:
        img (numpy.ndarray): 输入图像。
        window_size (int): 窗口大小。
    Returns:
        numpy.ndarray: 暗通道图像。
    """
    min_channel = np.min(img, axis=2)
    dark_channel = cv2.erode(min_channel, cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size)))
    return dark_channel


# 估计大气光
def estimate_atmospheric_light(img, dark_channel):
    """
    估计大气光。
    Args:
        img (numpy.ndarray): 输入图像。
        dark_channel (numpy.ndarray): 暗通道图像。
    Returns:
        numpy.ndarray: 大气光值。
    """
    num_pixels = int(0.001 * dark_channel.size)
    indices = np.unravel_index(np.argsort(dark_channel.ravel())[::-1][:num_pixels], dark_channel.shape)
    atmospheric_light = np.mean(img[indices], axis=0)
    return atmospheric_light


# 估计透射率
def estimate_transmission(img, atmospheric_light, omega=0.95):
    """
    估计透射率。
    Args:
        img (numpy.ndarray): 输入图像。
        atmospheric_light (numpy.ndarray): 大气光值。
        omega (float): 调节参数，通常取0.95。
    Returns:
        numpy.ndarray: 透射率图像。
    """
    normalized_img = img / atmospheric_light
    transmission = 1 - omega * get_dark_channel(normalized_img, window_size=15)
    return transmission


# 透射率细化
def refine_transmission(transmission, img):
    """
    使用引导滤波细化透射率。
    Args:
        transmission (numpy.ndarray): 原始透射率。
        img (numpy.ndarray): 原始图像，用作引导滤波的引导图。
    Returns:
        numpy.ndarray: 细化后的透射率。
    """
    # 将图像转换为灰度
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # 使用引导滤波进行透射率细化
    refined_transmission = cv2.ximgproc.guidedFilter(
        guide=gray.astype(np.float32) / 255.0,
        src=transmission.astype(np.float32),
        radius=15,
        eps=1e-3
    )
    return refined_transmission


# 恢复场景辐射
def recover_scene_radiance(img, atmospheric_light, transmission, t0=0.1):
    """
    恢复场景辐射。
    Args:
        img (numpy.ndarray): 输入图像。
        atmospheric_light (numpy.ndarray): 大气光值。
        transmission (numpy.ndarray): 透射率图像。
        t0 (float): 最小透射率，避免除零。
    Returns:
        numpy.ndarray: 恢复后的图像。
    """
    transmission = np.maximum(transmission, t0)
    radiance = (img - atmospheric_light) / transmission[..., None] + atmospheric_light
    return np.clip(radiance, 0, 1)


def histogram_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    使用自适应直方图均衡化增强图像。
    参数:
        img (numpy.ndarray): 输入图像。
        clip_limit (float): 对比度限制参数。
        tile_grid_size (tuple): 网格大小，控制局部均衡的范围。
    返回:
        numpy.ndarray: 均衡化后的图像。
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img


# 主函数：处理图像并计算指标
# 主函数：处理图像并计算指标
def enhance_and_evaluate(image_folder, output_excel, output_image_folder):
    os.makedirs(output_image_folder, exist_ok=True)
    results = []

    # 列出图像文件
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.bmp'))]

    for file_name in tqdm(image_files, desc="处理图像"):
        file_path = os.path.join(image_folder, file_name)
        if not os.path.isfile(file_path):
            print(f"跳过非文件: {file_name}")
            continue

        # 读取图像
        img = cv2.imread(file_path)
        if img is None:
            print(f"无法处理图像: {file_name}")
            continue

        reference = np.full_like(img, 128, dtype=np.uint8)

        # 原始指标
        psnr_before = calculate_psnr(img, reference)
        uciqe_before = calculate_uciqe(img)
        uiqm_before = calculate_uiqm(img)

        # 增强图像
        enhanced_img = enhance_underwater_image(img)

        # 如果增强后的图像与原始图像尺寸不一致，调整尺寸
        if img.shape != enhanced_img.shape:
            enhanced_img = cv2.resize(enhanced_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

        # 拼接图像
        comparison_img = np.hstack((img, enhanced_img))

        # 增强后的指标
        psnr_after = calculate_psnr(enhanced_img, reference)
        uciqe_after = calculate_uciqe(enhanced_img)
        uiqm_after = calculate_uiqm(enhanced_img)

        # 保存增强后的图像
        # enhanced_img_name = os.path.join(output_image_folder, f"enhanced_{file_name}")
        comparison_img_name = os.path.join(output_image_folder, f"comparison_{file_name}")

        try:
            # cv2.imwrite(enhanced_img_name, enhanced_img)
            cv2.imwrite(comparison_img_name, comparison_img)  # 保存拼接图像
        except Exception as e:
            print(f"保存图像失败: {file_name}, 错误: {e}")
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


# 主程序执行
if __name__ == "__main__":
    image_folder = r"../../data/附件一"  # 替换为你的输入文件夹路径
    output_excel = r"../../results/metrics/T4_optimizing_basedT3.xlsx"  # 替换为你的输出 Excel 文件路径
    output_image_folder = r"../../results/T4enhanced_optimizing_basedT3"  # 替换为你的输出图像文件夹
    enhance_and_evaluate(image_folder, output_excel, output_image_folder)
