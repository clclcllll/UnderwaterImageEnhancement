import cv2
import numpy as np
import os


def colors(image):
    """
    计算图像的颜色通道平均值（蓝、绿、红）。
    Args:
        image (numpy.ndarray): 输入图像。
    Returns:
        tuple: 蓝、绿、红通道的平均值。
    """
    mean_colors = cv2.mean(image)[:3]
    return mean_colors  # 返回 (blue, green, red)


def brightness(gray_image):
    """
    计算灰度图像的平均亮度。
    Args:
        gray_image (numpy.ndarray): 灰度图像。
    Returns:
        float: 平均亮度。
    """
    return np.mean(gray_image)


def variance(gray_image):
    """
    使用拉普拉斯算子计算图像清晰度（拉普拉斯方差）。
    Args:
        gray_image (numpy.ndarray): 灰度图像。
    Returns:
        float: 拉普拉斯方差。
    """
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


if __name__ == "__main__":
    # 定义输入和输出文件夹路径
    input_folder = "../../data/raw"
    output_folder = "../../data/processed"

    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历 raw 文件夹中的所有图像
    for index, filename in enumerate(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, filename)

        # 加载图像
        image = cv2.imread(file_path)
        if image is None:
            print(f"无法加载图像: {file_path}")
            continue

        # 转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 调用函数计算各指标
        blue, green, red = colors(image)
        mean_brightness = brightness(gray_image)
        laplacian_var = variance(gray_image)

        # 生成新的文件名
        new_filename = f"{index + 1}_{blue:.2f}_{green:.2f}_{red:.2f}_{mean_brightness:.2f}_{laplacian_var:.2f}.png"

        # 保存到 processed 文件夹
        output_path = os.path.join(output_folder, new_filename)
        cv2.imwrite(output_path, image)

        # 打印处理信息
        print(f"已处理: {filename} -> {new_filename}")

    print("\n所有图像处理完成！")
