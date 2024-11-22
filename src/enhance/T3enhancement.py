# enhancement.py

"""
T3 增强算法（物理模型）
该模块包含使用物理模型方法增强水下图像的函数。
"""

import numpy as np
import cv2

from src.enhance.enhancement_extra import super_resolution, sharpen_with_kernel


def enhance_underwater_image(img):
    """
    增强水下图像的主函数。
    包括白平衡、去雾、直方图均衡化、Gamma 校正，并可选进行超分辨率与锐化处理。

    Args:
        img (numpy.ndarray): 输入图像。
        use_super_resolution (bool): 是否进行超分辨率处理。
        use_sharpening (bool): 是否进行锐化处理。

    Returns:
        numpy.ndarray: 增强后的图像。
    """
    # # 可选：超分辨率处理
    # img = super_resolution(img)


    # 白平衡调整
    img = white_balance(img)

    #
    # # 可选：锐化处理
    # img = sharpen_with_kernel(img)


    # 去雾
    img = dehaze(img)

    # 降噪处理
    img = denoise(img)

    # 直方图均衡化
    img = histogram_equalization(img, clip_limit=1.5, tile_grid_size=(8, 8))

    # Gamma 校正
    img = gamma_correction(img)

    enhanced_img = img
    return enhanced_img

# 增加降噪处理
# 在增强过程中，加入降噪处理，使用非局部均值滤波器。
def denoise(img):
    """
    对图像进行降噪处理，减少噪点。
    """
    img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    return img


# 改进的白平衡函数
#调整后的 white_balance 函数
def white_balance(img):
    """
    改进的白平衡算法，避免过度校正。
    """
    img = img.astype(np.float32) / 255.0

    # 计算每个通道的平均值
    mean_r = np.mean(img[:, :, 2])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 0])

    # 计算校正因子，限制在合理范围内
    mean_gray = (mean_r + mean_g + mean_b) / 3
    correction_r = mean_gray / (mean_r + 1e-6)
    correction_g = mean_gray / (mean_g + 1e-6)
    correction_b = mean_gray / (mean_b + 1e-6)

    # 限制校正因子的范围，避免过度校正
    correction_r = np.clip(correction_r, 0.8, 1.2)
    correction_g = np.clip(correction_g, 0.8, 1.2)
    correction_b = np.clip(correction_b, 0.8, 1.2)

    # 应用校正因子
    img[:, :, 2] *= correction_r
    img[:, :, 1] *= correction_g
    img[:, :, 0] *= correction_b

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)



# 改进的去雾函数
def dehaze(img):
    """
    改进的去雾方法，避免过度增强亮度。
    """
    img = img.astype(np.float32) / 255.0

    # 计算暗通道图像
    dark_channel = get_dark_channel(img, window_size=15)

    # 调整大气光估计
    atmospheric_light = estimate_atmospheric_light(img, dark_channel)

    # 计算透射率，调整 omega
    omega = 0.9  # 降低去雾强度，避免过度增强
    transmission = estimate_transmission(img, atmospheric_light, omega)

    # 调整最小透射率 t0
    t0 = 0.2  # 增大 t0，防止过度增强
    transmission = np.maximum(transmission, t0)

    # 透射率细化
    transmission = refine_transmission(transmission, img)

    # 恢复场景辐射
    result = recover_scene_radiance(img, atmospheric_light, transmission)

    return (result * 255).astype(np.uint8)



def get_dark_channel(img, window_size):
    """
    计算暗通道图像。
    """
    min_channel = np.min(img, axis=2)
    dark_channel = cv2.erode(min_channel, cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size)))
    return dark_channel


def estimate_atmospheric_light(img, dark_channel):
    """
    估计大气光。
    """
    num_pixels = int(0.001 * dark_channel.size)
    indices = np.unravel_index(np.argsort(dark_channel.ravel())[::-1][:num_pixels], dark_channel.shape)
    atmospheric_light = np.mean(img[indices], axis=0)
    return atmospheric_light


def estimate_transmission(img, atmospheric_light, omega=0.95):
    """
    估计透射率。
    """
    normalized_img = img / atmospheric_light
    transmission = 1 - omega * get_dark_channel(normalized_img, window_size=15)
    return transmission


def refine_transmission(transmission, img):
    """
    使用引导滤波细化透射率。
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


def recover_scene_radiance(img, atmospheric_light, transmission, t0=0.1):
    """
    恢复场景辐射。
    """
    transmission = np.maximum(transmission, t0)
    radiance = (img - atmospheric_light) / transmission[..., None] + atmospheric_light
    return np.clip(radiance, 0, 1)


# 改进的 Gamma 校正函数
def gamma_correction(img, gamma=None):
    """
    动态调整 Gamma 值，避免过度校正。
    """
    img = img.astype(np.float32) / 255.0

    # 根据图像平均亮度动态调整 Gamma 值
    mean_brightness = np.mean(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)) / 255.0

    if gamma is None:
        if mean_brightness < 0.5:
            gamma = 1.5
        else:
            gamma = 1.0

    # 限制 Gamma 值范围
    gamma = np.clip(gamma, 0.8, 1.5)

    img = np.power(img, gamma)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)



# 自适应直方图均衡化（CLAHE）
def histogram_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对RGB图像进行自适应直方图均衡化（CLAHE），限制亮处过亮。
    """
    # 将图像转换为YUV色彩空间
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 对Y通道（亮度）应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

    # 转回BGR色彩空间
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img
