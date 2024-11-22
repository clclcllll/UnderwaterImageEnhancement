# T3.py
# T3.py： 负责遍历指定路径下的所有图像，调用增强算法生成增强后的图像，并将结果（图像文件名、PSNR、UCIQE、UIQM）输出到 T3.csv 文件。
import os
import cv2
import pandas as pd
from src.utils.evaluation import calculate_psnr, calculate_uciqe, calculate_uiqm
from src.enhance.T3enhancement import enhance_underwater_image


# 主程序
# 输入和输出文件夹路径
input_folder_path = '../../data/附件一'  # 替换为你的测试图像文件夹路径
output_folder_path = '../../results/T3enhanced'  # 替换为你想要的输出图像文件夹路径

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 结果 CSV 文件路径
result_file_path = '../../results/metrics/plots/T3.csv'

# 读取附件一测试图像（支持 jpg 和 png 格式以及其他格式）
files = [f for f in os.listdir(input_folder_path) if f.endswith(('.jpg', '.png'))]
num_images = len(files)

# 初始化结果列表
results = []

# 处理每个图像文件
for i, file_name in enumerate(files):
    print(f'正在处理图像 {file_name}')
    image_path = os.path.join(input_folder_path, file_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f'无法读取图像 {file_name}')
        continue

    # 原始图像
    original_img = img.copy()

    # 增强图像
    enhanced_img = enhance_underwater_image(img)

    # 确保增强后的图像尺寸与原始图像一致
    if enhanced_img.shape != original_img.shape:
        enhanced_img = cv2.resize(enhanced_img, (original_img.shape[1], original_img.shape[0]))

    # 计算评估指标
    psnr = calculate_psnr(original_img, enhanced_img)
    uciqe = calculate_uciqe(enhanced_img)
    uiqm = calculate_uiqm(enhanced_img)

    # 存储结果
    results.append({
        'image file name': file_name,
        'PSNR': psnr,
        'UCIQE': uciqe,
        'UIQM': uiqm
    })

    # 保存增强后的图像，添加 'enhanced_' 前缀
    enhanced_image_path = os.path.join(output_folder_path, 'enhanced_' + file_name)
    cv2.imwrite(enhanced_image_path, enhanced_img)

    # 打印进度
    print(f'已处理 {i + 1} / {num_images} 张图像。')

# 将结果保存到 CSV 文件
df = pd.DataFrame(results)
df.to_csv(result_file_path, index=False)
