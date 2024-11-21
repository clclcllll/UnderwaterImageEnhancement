import cv2
from src.analysis.evaluation import calculate_psnr, calculate_uciqe, calculate_uiqm

enhanced_image = cv2.imread("data/processed/153_71.64_52.35_34.31_49.17_41.05.png")

# 计算 UCIQE
uciqe_value = calculate_uciqe(enhanced_image)
print(f"UCIQE: {uciqe_value:.2f}")

# 计算 UIQM
uiqm_value = calculate_uiqm(enhanced_image)
print(f"UIQM: {uiqm_value:.2f}")
