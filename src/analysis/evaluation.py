import cv2
import numpy as np
import os
import pandas as pd

# 增强水下图像的函数
def enhance_underwater_image(img):
    img = white_balance(img)
    img = dehaze(img)
    img = histogram_equalization(img)
    img = gamma_correction(img, 2.2)
    return img

# 白平衡调整
def white_balance(img):
    R, G, B = cv2.split(img.astype(np.float32))
    meanR, meanG, meanB = np.mean(R), np.mean(G), np.mean(B)
    total_mean = (meanR + meanG + meanB) / 3
    R, G, B = R * (total_mean / meanR), G * (total_mean / meanG), B * (total_mean / meanB)
    img = cv2.merge([R, G, B])
    return np.clip(img, 0, 255).astype(np.uint8)

# 去雾
def dehaze(img):
    dark_channel = dark_channel_prior(img)
    atmospheric_light = estimate_atmospheric_light(img, dark_channel)
    transmission = estimate_transmission(img, atmospheric_light)
    transmission = refine_transmission(transmission)
    img = recover_scene_radiance(img, atmospheric_light, transmission)
    return np.clip(img, 0, 255).astype(np.uint8)

# 暗通道先验
def dark_channel_prior(img, window_size=15):
    dark_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(dark_channel, kernel)
    return dark_channel

# 估计大气光
def estimate_atmospheric_light(img, dark_channel):
    h, w = dark_channel.shape
    n_pixels = h * w
    n_brightest = max(n_pixels // 1000, 1)
    brightest_pixels = np.argsort(dark_channel.ravel())[-n_brightest:]
    return np.max(img.reshape(-1, 3)[brightest_pixels], axis=0)

# 估计透射率
def estimate_transmission(img, atmospheric_light, omega=0.95):
    img_normalized = img / atmospheric_light
    transmission = 1 - omega * np.min(img_normalized, axis=2)
    return transmission

# 透射率细化
def refine_transmission(transmission):
    transmission = cv2.GaussianBlur(transmission, (15, 15), 1.5)
    return transmission

# 恢复场景辐射
def recover_scene_radiance(img, atmospheric_light, transmission, t0=0.1):
    transmission = np.maximum(transmission, t0)
    img = (img - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light
    return img

# 直方图均衡化
def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

# Gamma校正
def gamma_correction(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)

# 计算 PSNR
def calculate_psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    max_val = 255
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr

# 计算 UCIQE
def calculate_uciqe(img):
    img = img.astype(np.float32) / 255.0
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    chroma = np.sqrt(np.sum((img - np.mean(img, axis=(0, 1))) ** 2, axis=2))
    uc = np.mean(chroma)
    sc = np.std(chroma)
    l = np.mean(img[:, :, 0])
    conl = np.max(img[:, :, 0]) - np.min(img[:, :, 0])
    us = np.mean(chroma / (img[:, :, 0] + 1e-6))
    return c1 * sc + c2 * conl + c3 * us

# 计算 UIQM
def calculate_uiqm(img):
    img = img.astype(np.float32) / 255.0
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm = np.std(img[:, :, 0] - img[:, :, 1])
    uism = np.mean(cv2.Sobel(img, cv2.CV_64F, 1, 1))
    uiconm = np.mean(img)
    return c1 * uicm + c2 * uism + c3 * uiconm
