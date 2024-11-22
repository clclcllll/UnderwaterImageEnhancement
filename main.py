import cv2
import numpy as np

from src.utils.evaluation import calculate_uciqe, calculate_uiqm

enhanced_image = cv2.imread("data/T3enhanced/92_121.33_108.26_28.21_85.81_7340.52.png")
normalized_image = enhanced_image.astype(np.float32) / 255.0

# 计算 UCIQEzheg
uciqe_value = calculate_uciqe(enhanced_image)
print(f"UCIQE: {uciqe_value:.2f}")

# 计算 UIQM
uiqm_value = calculate_uiqm(enhanced_image)
print(f"UIQM: {uiqm_value:.2f}")
