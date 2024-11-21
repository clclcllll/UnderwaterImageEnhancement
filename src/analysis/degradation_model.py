# degradation_model.py：用于构建和模拟水下图像退化模型的脚本

"""
simulate_color_cast： 模拟偏色现象，通过对每个颜色通道应用不同的透射率和环境光。
simulate_low_light： 模拟弱光现象，对整个图像应用统一的透射率和环境光。
simulate_blur： 模拟模糊现象，使用高斯模糊滤波器。
测试函数： 在主程序中读取示例图像，分别应用三种退化模拟函数，并将结果保存到 results/enhanced_images/ 文件夹中。
"""

# 导入必要的库
import cv2
import numpy as np


# 定义模拟偏色的函数
def simulate_color_cast(image, t_r, t_g, t_b, B_r, B_g, B_b):
    """
    模拟偏色现象

    参数：
    - image: 输入图像（BGR格式）
    - t_r, t_g, t_b: 红、绿、蓝通道的透射率
    - B_r, B_g, B_b: 红、绿、蓝通道的环境光值

    返回值：
    - degraded_image: 退化后的图像
    """
    # 将图像转换为浮点型
    image = image.astype(np.float32)
    # 分离颜色通道
    B_channel, G_channel, R_channel = cv2.split(image)
    # 应用透射率和环境光
    R_channel = R_channel * t_r + B_r * (1 - t_r)
    G_channel = G_channel * t_g + B_g * (1 - t_g)
    B_channel = B_channel * t_b + B_b * (1 - t_b)
    # 合并通道
    degraded_image = cv2.merge([B_channel, G_channel, R_channel])
    # 将图像裁剪到0-255并转换为8位无符号整数
    degraded_image = np.clip(degraded_image, 0, 255).astype(np.uint8)
    return degraded_image


# 定义模拟弱光的函数
def simulate_low_light(image, t, B):
    """
    模拟弱光现象

    参数：
    - image: 输入图像（BGR格式）
    - t: 透射率（标量）
    - B: 环境光（标量）

    返回值：
    - degraded_image: 退化后的图像
    """
    # 将图像转换为浮点型
    image = image.astype(np.float32)
    # 应用透射率和环境光
    degraded_image = image * t + B * (1 - t)
    # 将图像裁剪到0-255并转换为8位无符号整数
    degraded_image = np.clip(degraded_image, 0, 255).astype(np.uint8)
    return degraded_image


# 定义模拟模糊的函数
def simulate_blur(image, kernel_size):
    """
    模拟模糊现象

    参数：
    - image: 输入图像（BGR格式）
    - kernel_size: 高斯模糊的核大小（奇数）

    返回值：
    - blurred_image: 模糊后的图像
    """
    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image


# 测试函数
if __name__ == '__main__':
    # 读取示例图像
    image_path = '../../data/raw/3_img_.png'  # 根据实际路径和文件名调整
    image = cv2.imread(image_path)

    # 模拟偏色
    t_r, t_g, t_b = 0.6, 0.8, 0.9  # 设定透射率
    B_r, B_g, B_b = 10, 10, 10  # 设定环境光
    color_cast_image = simulate_color_cast(image, t_r, t_g, t_b, B_r, B_g, B_b)
    cv2.imwrite('../../results/enhanced_images/color_cast_image.png', color_cast_image)

    # 模拟弱光
    t = 0.5  # 设定透射率
    B = 5  # 设定环境光
    low_light_image = simulate_low_light(image, t, B)
    cv2.imwrite('../../results/enhanced_images/low_light_image.png', low_light_image)

    # 模拟模糊
    kernel_size = 15  # 设定核大小
    blurred_image = simulate_blur(image, kernel_size)
    cv2.imwrite('../../results/enhanced_images/blurred_image.png', blurred_image)
