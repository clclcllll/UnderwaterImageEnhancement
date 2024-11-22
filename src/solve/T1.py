# 主程序
import os
import glob
import cv2
import numpy as np
import pandas as pd


def analyzeColorBias(img):
    # 将图像转换为范围 [0,1] 的浮点数
    img_float = img.astype(np.float32) / 255.0

    # 分离图像的 B、G、R 通道
    B, G, R = cv2.split(img_float)

    # 计算每个通道的均值
    meanR = np.mean(R)
    meanG = np.mean(G)
    meanB = np.mean(B)

    # 判断图像是否存在颜色偏差
    isColorBiased = ((meanB > meanR + 0.05) and (meanB > meanG + 0.05)) or \
                    ((meanG > meanB + 0.05) and (meanG > meanR + 0.05)) or \
                    ((meanR > meanB + 0.05) and (meanR > meanG + 0.05))
    return isColorBiased


def analyzeLowLight(img):
    # 将图像转换为灰度图像
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 将灰度图像转换为 [0,1] 的浮点数
    grayImg_float = grayImg.astype(np.float32) / 255.0

    # 计算灰度图像的平均亮度
    meanBrightness = np.mean(grayImg_float)

    # 判断图像是否为低光
    threshold = 0.3  # 亮度阈值
    isLowLight = meanBrightness < threshold
    return isLowLight


def analyzeBlur(img):
    # 将图像转换为灰度图像
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算图像的拉普拉斯变换
    laplacian = cv2.Laplacian(grayImg, cv2.CV_64F)

    # 将拉普拉斯结果展平为一维数组
    linearizedImg = laplacian.flatten()

    # 计算拉普拉斯变换结果的方差
    laplacianVar = np.var(linearizedImg)

    # 判断图像是否模糊
    threshold = 60  # 方差阈值
    isBlurry = laplacianVar < threshold
    return isBlurry


def analyzeImageDataset(folderPath, resultFilePath):
    # 获取文件夹中的所有文件，并按文件在目录中的顺序处理
    entries = sorted(os.scandir(folderPath), key=lambda e: e.name)

    # 初始化结果列表
    results = []

    # 遍历每个图像文件
    for i, entry in enumerate(entries):
        if entry.is_file() and (entry.name.lower().endswith('.jpg') or entry.name.lower().endswith('.png')):
            fileName = entry.name
            imagePath = entry.path
            print(f'正在处理图像 {fileName}')
            img = cv2.imread(imagePath)

            # 检查图像是否成功加载
            if img is None:
                print(f'无法读取图像 {fileName}')
                continue

            # 分析颜色偏差
            isColorBiased = analyzeColorBias(img)

            # 分析低光
            isLowLight = analyzeLowLight(img)

            # 分析模糊
            isBlurry = analyzeBlur(img)

            # 将结果添加到列表
            results.append([fileName, isColorBiased, isLowLight, isBlurry])

            # 打印进度
            print(f'已处理 {i + 1} / {len(entries)} 张图像')

    # 将结果保存为 CSV 文件
    df = pd.DataFrame(results, columns=['文件名', '是否颜色偏差', '是否低光', '是否模糊'])
    df.to_csv(resultFilePath, index=False)


if __name__ == "__main__":
    folderPath = r'../../data/附件一/'  # 替换图像文件夹路径
    resultFilePath = '../../results/metrics/plots/T1.csv'  # 结果文件路径

    analyzeImageDataset(folderPath, resultFilePath)

