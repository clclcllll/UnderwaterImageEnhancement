"""
evaluation.py
水下图像增强的评价指标
该模块包含计算 PSNR、UCIQE 和 UIQM 指标的函数。
"""

import numpy as np
import cv2
from skimage import metrics

def calculate_psnr(original, enhanced):
    print('\n[PSNR 计算]')
    mse = np.mean((original - enhanced) ** 2)
    print(f'MSE: {mse}')
    psnr = metrics.peak_signal_noise_ratio(original, enhanced, data_range=255)
    print(f'计算得到的 PSNR: {psnr}')
    return psnr

def calculate_uciqe(img):
    print('\n[UCIQE 计算]')
    # 确保输入图像为 uint8 类型
    img = img.astype('uint8')
    C1 = 0.4680
    C2 = 0.2745
    C3 = 0.2576

    # 转换到 LAB 颜色空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0

    # 计算色度
    chroma = np.sqrt(a ** 2 + b ** 2)
    chroma_std = np.std(chroma)
    print(f'色度标准差 (chroma_std): {chroma_std}')

    # 计算亮度对比度
    l_contrast = np.max(l) - np.min(l)
    print(f'亮度对比度 (l_contrast): {l_contrast}')

    # 转换到 HSV 颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    # 计算饱和度均值
    saturation_mean = np.mean(saturation)
    print(f'饱和度均值 (saturation_mean): {saturation_mean}')

    # 计算最终 UCIQE
    uciqe = C1 * chroma_std + C2 * l_contrast + C3 * saturation_mean
    print(f'计算得到的 UCIQE: {uciqe}')
    return uciqe


def calculate_uiqm(img):
    print('\n[UIQM 计算]')
    img = img.astype('float32') / 255.0
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753

    uicm = calculate_uicm(img)
    print(f'UICM: {uicm}')
    uism = calculate_uism(img)
    print(f'UISM: {uism}')
    uiconm = calculate_uiconm(img)
    print(f'UICONM: {uiconm}')

    uiqm = c1 * uicm + c2 * uism + c3 * uiconm
    print(f'计算得到的 UIQM: {uiqm}')
    return uiqm

def calculate_uicm(img):
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]
    rg = r - g
    yb = (r + g) / 2 - b
    rg_std = np.std(rg)
    yb_std = np.std(yb)
    uicm = -0.0268 * rg_std + -0.1586 * yb_std
    print(f'rg_std: {rg_std}, yb_std: {yb_std}')
    return uicm

def calculate_uism(img):
    gray = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
    uism = np.mean(gradient_magnitude)
    return uism

def calculate_uiconm(img):
    h, w, _ = img.shape
    N = h * w
    alpha = 0.1  # 权重因子
    chroma = np.sqrt(np.sum((img - np.mean(img, axis=(0, 1))) ** 2, axis=2))
    uiconm = alpha * np.sum(chroma) / N
    return uiconm
