# enhancement_extra.py

"""
额外的增强模块，包括超分辨率和锐化处理。
"""

import numpy as np
import cv2


def super_resolution(img):
    """
    使用预训练的超分辨率模型提高分辨率。
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    # 加载预训练的超分辨率模型，路径需要替换为实际模型路径
    model_path = "../solve/ESPCN_x2.pb"  # 替换为实际模型路径
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
    """
    # 定义轻微锐化卷积核
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])  # 减少中心值，降低锐化强度

    # 应用卷积核
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img
