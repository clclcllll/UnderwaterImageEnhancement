import io
import math

import cv2
import numpy as np
from skimage import color, filters


# 1. 计算 PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(original, enhanced):
    """
    计算 PSNR 值，用于评估增强图像与原始图像的相似性。
    :param original: 原始图像 (numpy array)
    :param enhanced: 增强后的图像 (numpy array)
    :return: PSNR 值 (float)
    """
    mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32)) ** 2)
    if mse == 0:  # 避免 log(0)
        return float('inf')
    max_pixel = 255.0  # 假设像素值范围为 0-255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def img_loader(path):
    img = cv2.imread(path)
    return img


def psnr1(image_true, image_test):
    '''image_true: ground_truth图像
       image_test: 恢复后的图像
       psnr1和psnr2 区别在于求mse的时候是否进行255归一化,如果先对图像进行归一化,那么后面MAX=1, 否则为255
    '''
    mse = np.mean((image_true / 1.0 - image_test / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(255 / math.sqrt(mse))
    return psnr


def psnr2(image_true, image_test):
    '''image_true: ground_truth图像
       image_test: 恢复后的图像
       psnr1和psnr2 区别在于求mse的时候是否进行255归一化,如果先对图像进行归一化,那么后面MAX=1, 否则为255
    '''
    mse = np.mean((image_true / 255.0 - image_test / 255.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr

def calculate_uciqe(image):
    """
    计算水下图像的 UCIQE 值
    :param image: 输入的 RGB 图像（值在 [0, 1] 范围内）
    :return: 计算得到的 UCIQE 值
    """
    # 归一化
    image = image / 255.0

    # 权重系数
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576

    # 将 RGB 图像转换为 LAB 颜色空间
    lab_image = color.rgb2lab(image)
    L = lab_image[:, :, 0]  # 明度 (L)
    A = lab_image[:, :, 1]  # 色度分量 a
    B = lab_image[:, :, 2]  # 色度分量 b

    # 1. 计算色度 (Chroma)
    chroma = np.sqrt(A**2 + B**2)
    sigma_c = np.std(chroma)  # 色度的标准差

    # 2. 计算亮度对比度 (Contrast of luminance)
    top_percent = int(0.01 * L.size)  # 选取亮度的前1%
    sorted_L = np.sort(L.flatten())
    contrast_l = np.mean(sorted_L[-top_percent:]) - np.mean(sorted_L[:top_percent])

    # 3. 计算饱和度 (Saturation)
    chroma_flat = chroma.flatten()
    L_flat = L.flatten()
    saturation = np.divide(chroma_flat, L_flat, out=np.zeros_like(chroma_flat), where=L_flat != 0)
    mean_saturation = np.mean(saturation)

    # 计算 UCIQE
    uciqe = c1 * sigma_c + c2 * contrast_l + c3 * mean_saturation

    return uciqe


def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/abstract/document/5609219
    """
    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    # weight
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    k1 = int(k1)
    k2 = int(k2)
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w*val

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)


def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)


def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag

def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    # get eme for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)


def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    # weight
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
    val = 0
    k1 = int(k1)
    k2 = int(k2)
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            # bound checks, can't do log(0)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val



def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    ### UCIQE: https://ieeexplore.ieee.org/abstract/document/7300447
    #c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### UIQM https://ieeexplore.ieee.org/abstract/document/7305804
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm