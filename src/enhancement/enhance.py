import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure
from tqdm import tqdm
from src.analysis import classify


# 图像增强模块
def solve_low_resolution(img, scale=2):
    """
    解决低分辨率问题，包括分辨率增强和锐化处理。
    Args:
        img (numpy.ndarray): 输入图像，BGR格式。
        scale (int): 分辨率放大的倍数。
    Returns:
        numpy.ndarray: 分辨率增强后的图像。
    """
    # 使用双三次插值进行分辨率增强
    height, width = img.shape[:2]
    enhanced_img = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

    # 应用锐化滤波器增强细节
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])  # 拉普拉斯锐化核
    enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)

    return enhanced_img

def super_resolution(img):
    """
    使用预训练的超分辨率模型提高分辨率。
    Args:
        img (numpy.ndarray): 输入低分辨率图像。
    Returns:
        numpy.ndarray: 高分辨率图像。
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    # 加载预训练的 ESPCN 模型
    model_path = "../../export/ESPCN_x2.pb"  # 替换为实际模型路径
    try:
        sr.readModel(model_path)
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")

    # 设置模型和放大倍数
    sr.setModel("espcn", 2)  # 使用 ESPCN 模型，放大倍数为 2

    # 确保输入为 RGB 格式
    if len(img.shape) == 2:  # 如果是灰度图像，转为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:  # BGR 转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("输入图像必须是灰度图像或 3 通道彩色图像。")

    # 调整输入尺寸为 2 的倍数
    height, width = img.shape[:2]
    new_height = (height // 2) * 2
    new_width = (width // 2) * 2
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 执行超分辨率
    try:
        high_res_img = sr.upsample(resized_img)
    except Exception as e:
        raise RuntimeError(f"超分辨率失败: {e}")

    # 转换回 BGR 格式以保持一致性
    high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_RGB2BGR)

    return high_res_img

def sharpen_with_kernel(img):
    """
    使用卷积核锐化图像。
    Args:
        img (numpy.ndarray): 输入图像，BGR格式。
    Returns:
        numpy.ndarray: 锐化后的图像。
    """
    # 定义轻微锐化卷积核
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])  # 减少中心值，降低锐化强度

    # 应用卷积核
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img



def enhance_underwater_image(img):
    """
    水下图像增强，包括白平衡调整、去雾、直方图均衡化和伽马校正。
    Args:
        img (numpy.ndarray): 输入图像，BGR格式。
    Returns:
        numpy.ndarray: 增强后的图像。
    """

    img = super_resolution(img)

    # 转换为灰度图以计算亮度和清晰度
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取颜色通道平均值、亮度和拉普拉斯方差
    blue, green, red = classify.colors(img)
    mean_brightness = classify.brightness(gray_image)

    img = sharpen_with_kernel(img)

    # 去雾（始终应用）
    img = dehaze(img)

    # 伽马校正（仅在平均亮度过低时应用）
    if mean_brightness < 50:  # 根据图像亮度动态调整
        img = gamma_correction(img, gamma=1.8)

    # 白平衡调整（适配水下图像）
    if abs(blue - green) > 10 or abs(blue - red) > 10 or abs(green - red) > 10:
        img = white_balance(img)

    # 自适应直方图均衡化
    img = histogram_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8))

    return img


# 白平衡调整
def white_balance(img):
    """
    使用改进的白平衡算法调整图像色彩，避免偏色或过度调整，特别适配水下图像。
    Args:
        img (numpy.ndarray): 输入图像，BGR格式。
    Returns:
        numpy.ndarray: 调整后的图像。
    """
    # 将图像转换为浮点类型并归一化到[0, 1]
    img = img.astype(np.float32) / 255.0

    # 计算每个通道的均值（避免极值影响）
    mean_r = np.mean(img[:, :, 2][img[:, :, 2] > 0.01])  # 排除过暗像素
    mean_g = np.mean(img[:, :, 1][img[:, :, 1] > 0.01])  # 排除过暗像素
    mean_b = np.mean(img[:, :, 0][img[:, :, 0] > 0.01])  # 排除过暗像素

    # 计算整体灰度的平均值（灰世界假设）
    mean_gray = (mean_r + mean_g + mean_b) / 3

    # 计算校正因子，限制红色因子以避免过度拉高
    correction_r = min(mean_gray / (mean_r + 1e-6), 1.5)  # 限制最大值
    correction_g = mean_gray / (mean_g + 1e-6)
    correction_b = mean_gray / (mean_b + 1e-6)

    # 如果绿色或蓝色主导场景，减少红色校正幅度
    if mean_g > mean_r and mean_b > mean_r:
        correction_r *= 0.85  # 适度降低红色调整比例

    # 应用校正因子调整每个通道
    img[:, :, 2] *= correction_r  # 调整红色通道
    img[:, :, 1] *= correction_g  # 调整绿色通道
    img[:, :, 0] *= correction_b  # 调整蓝色通道

    # 将像素值限制在[0, 1]范围，避免溢出
    img = np.clip(img, 0, 1)

    # 将图像转换回[0, 255]的8位整数格式
    return (img * 255).astype(np.uint8)


# 图像去雾模块
def dehaze(img):
    """
    改进的去雾方法，适配水下图像。
    Args:
        img (numpy.ndarray): 输入图像，BGR格式。
    Returns:
        numpy.ndarray: 去雾后的图像。
    """
    # 转为浮点类型并归一化
    img = img.astype(np.float32) / 255.0

    # 1. 计算暗通道图像
    dark_channel = get_dark_channel(img, window_size=15)

    # 2. 估计大气光
    atmospheric_light = estimate_atmospheric_light(img, dark_channel)

    # 3. 计算透射率
    omega = 0.95  # 调整去雾强度
    transmission = estimate_transmission(img, atmospheric_light, omega)

    # 4. 透射率细化
    transmission = refine_transmission(transmission, img)

    # 5. 恢复场景辐射
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


# 直方图均衡化
def histogram_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对RGB图像进行自适应直方图均衡化（CLAHE），限制亮处过亮。

    Args:
        img (numpy.ndarray): 输入图像。
        clip_limit (float): 对比度限制参数，越大增强效果越明显。
        tile_grid_size (tuple): 网格大小，控制局部均衡的范围。

    Returns:
        numpy.ndarray: 均衡化后的图像。
    """
    # 将图像转换为YUV色彩空间
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 对Y通道（亮度）应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

    # 转回BGR色彩空间
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
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


if __name__ == "__main__":
    def process_images_with_comparison(input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.bmp'))]

        for i, file in enumerate(tqdm(files, desc="Processing Images")):
            input_path = os.path.join(input_dir, file)
            enhanced_output_path = os.path.join(output_dir, f"{i + 1:03d}.png")  # 增强后的图片路径
            comparison_output_path = os.path.join(output_dir, f"{i + 1:03d}_comparison.png")  # 对比图片路径

            # 读取图像
            img = cv2.imread(input_path)

            # 增强并去雾
            enhanced_img = enhance_underwater_image(img)

            # 保存增强后的图像
            cv2.imwrite(enhanced_output_path, enhanced_img)

            # 拼接初始图像和增强后的图像
            # 如果增强后的图像分辨率发生变化，需要调整为与原图相同的大小
            if img.shape != enhanced_img.shape:
                enhanced_img = cv2.resize(enhanced_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

            # 水平拼接初始图像和增强后的图像
            comparison_img = np.hstack((img, enhanced_img))

            # 保存对比图像
            cv2.imwrite(comparison_output_path, comparison_img)


    # 指定输入和输出路径
    input_folder = r"C:\Users\DELL\Desktop\UnderwaterImageEnhancement\data\processed"
    output_folder = r"C:\Users\DELL\Desktop\UnderwaterImageEnhancement\results\enhanced_images"

    # 处理图像并保存对比图片
    process_images_with_comparison(input_folder, output_folder)
