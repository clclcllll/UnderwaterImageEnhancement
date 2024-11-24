import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


# 生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# 加载生成器模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

# generator.load_state_dict(torch.load('generator.pth'))
#  加载模型时，需要指定 map_location=torch.device('cpu')，否则可能会报错
generator.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))


generator.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# 计算指标
def calculate_psnr(image, reference):
    mse = np.mean((image.astype(np.float64) - reference.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_uciqe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    chroma = np.sqrt(a.astype(np.float64) ** 2 + b.astype(np.float64) ** 2)
    return np.std(chroma) + np.std(l) + np.mean(l) / (np.mean(chroma) + 1e-6)


def calculate_uiqm(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return 0.5 * np.std(h.astype(np.float64)) + 0.25 * np.std(s.astype(np.float64)) + 0.25 * np.mean(v.astype(np.float64))


# 处理图像并保存指标
def process_and_analyze(input_folder, output_folder, output_excel):
    os.makedirs(output_folder, exist_ok=True)
    results = []

    for filename in os.listdir(input_folder):
        if filename.endswith(('png', 'jpg', 'jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取原始图像
            original_image = Image.open(input_path).convert('RGB')
            original_image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

            # 参考图像（灰色背景，与生成图像尺寸匹配）
            reference = np.full((256, 256, 3), 128, dtype=np.uint8)

            # 原始图像指标
            psnr_before = calculate_psnr(original_image_bgr, cv2.resize(reference, (
                original_image_bgr.shape[1], original_image_bgr.shape[0])))

            uciqe_before = calculate_uciqe(original_image_bgr)
            uiqm_before = calculate_uiqm(original_image_bgr)

            # 生成图像
            image_tensor = transform(original_image).unsqueeze(0).to(device)
            with torch.no_grad():
                generated_image_tensor = generator(image_tensor)
            # 反归一化生成的图像
            generated_image_to_save = (generated_image_tensor + 1) / 2.0
            # 保存生成的图像（与原始代码一致）
            save_image(generated_image_to_save, output_path)
            print(f'Processed and saved {filename} to {output_path}')

            # 将生成的图像转换为 numpy 数组用于指标计算
            generated_image = (generated_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            generated_image_bgr = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)

            # 生成图像指标
            psnr_after = calculate_psnr(generated_image_bgr, reference)
            uciqe_after = calculate_uciqe(generated_image_bgr)
            uiqm_after = calculate_uiqm(generated_image_bgr)

            # 保存结果
            results.append({
                "image file name": filename,
                "Degraded Image Classification": None,
                "PSNR": psnr_before,
                "UCIQE": uciqe_before,
                "UIQM": uiqm_before,
                "PSNR-IM": psnr_after,
                "UCIQE-IM": uciqe_after,
                "UIQM-IM": uiqm_after,
            })

    # 保存到 Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"Results saved to {output_excel}")


# 主函数
if __name__ == "__main__":
    input_folder = '../../data/附件一'
    output_folder = '../../results/T4enhanced_basedCNN'
    output_excel = '../../results/metrics/T4_basedCNN.xlsx'

    process_and_analyze(input_folder, output_folder, output_excel)
