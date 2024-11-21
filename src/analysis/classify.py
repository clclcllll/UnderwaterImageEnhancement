import cv2
import numpy as np
import os

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

    # 计算颜色通道的平均值（蓝、绿、红）
    mean_colors = cv2.mean(image)[:3]
    blue, green, red = mean_colors

    # 计算平均亮度
    mean_brightness = np.mean(gray_image)

    # 使用拉普拉斯算子计算清晰度（拉普拉斯方差）
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    # 生成新的文件名
    # 序号_蓝色值_绿色值_红色值_平均亮度_拉普拉斯方差.png
    new_filename = f"{index+1}_{blue:.2f}_{green:.2f}_{red:.2f}_{mean_brightness:.2f}_{laplacian_var:.2f}.png"

    # 保存到 processed 文件夹
    output_path = os.path.join(output_folder, new_filename)
    cv2.imwrite(output_path, image)

    # 打印处理信息
    print(f"已处理: {filename} -> {new_filename}")

print("\n所有图像处理完成！")
