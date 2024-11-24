
import multiprocessing
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
from pytorch_msssim import ssim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image


# 配置类
class Config:
    def __init__(self):
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择 GPU 或 CPU

        # 训练参数
        self.batch_size = 8  # 批量大小
        self.num_epochs = 1  # 训练轮数
        self.lr = 0.0002  # 学习率

        # 数据路径
        self.data_dir = '../../data/附件一'  # 替换为实际的图像文件夹路径
        self.output_dir = '../../results/T4_generator'  # 保存生成图像的目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


# 数据集类
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if
                            file.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Encoder
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
            # Decoder
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


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)  # 将输出展平为一维向量


# 感知损失
def perceptual_loss(input, target, device):
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    input_features = vgg(input)
    target_features = vgg(target)
    return torch.mean((input_features - target_features) ** 2)


# SSIM损失
def ssim_loss(input, target):
    return 1 - ssim(input, target, data_range=1, size_average=True)


# 简单的扩散模型类
class SimpleDiffusionModel(nn.Module):
    def __init__(self, device):
        super(SimpleDiffusionModel, self).__init__()
        # 使用预训练的UNet模型
        self.unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
        self.scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

    def forward(self, x, timestep):
        # 传递输入图像和时间步
        return self.unet(x, timestep).sample



def main():
    # 设置多进程启动方式为 'fork'
    multiprocessing.set_start_method("fork", force=True)

    # 初始化配置
    config = Config()

    # 加载数据
    dataset = CustomImageDataset(config.data_dir, transform=config.transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 初始化模型
    generator = Generator().to(config.device)
    discriminator = Discriminator().to(config.device)
    diffusion_model = SimpleDiffusionModel(config.device).to(config.device)

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optimizer_diffusion = optim.Adam(diffusion_model.parameters(), lr=config.lr)

    # 训练
    for epoch in range(config.num_epochs):
        for i, images in enumerate(dataloader):
            images = images.to(config.device)

            # 训练判别器
            optimizer_D.zero_grad()
            real_output = discriminator(images)
            real_labels = torch.ones_like(real_output).to(config.device)
            fake_images = generator(images)
            fake_output = discriminator(fake_images.detach())
            fake_labels = torch.zeros_like(fake_output).to(config.device)
            d_loss_real = criterion(real_output, real_labels)
            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_images)
            g_loss_adversarial = criterion(fake_output, real_labels)
            g_loss_perceptual = perceptual_loss(fake_images, images, config.device)
            g_loss_ssim = ssim_loss(fake_images, images)

            # 使用扩散模型进行后处理
            with torch.no_grad():
                # 生成时间步
                timesteps = torch.randint(0, diffusion_model.scheduler.config.num_train_timesteps,
                                          (fake_images.size(0),),
                                          device=config.device).long()
                enhanced_images = diffusion_model(fake_images, timesteps)

            g_loss_diffusion = torch.mean((enhanced_images - images) ** 2)

            # 综合损失
            g_loss = g_loss_adversarial + 0.001 * g_loss_perceptual + 0.01 * g_loss_ssim + 0.005 * g_loss_diffusion
            g_loss.backward()
            optimizer_G.step()

            # 训练扩散模型
            optimizer_diffusion.zero_grad()
            # 生成时间步
            timesteps = torch.randint(0, diffusion_model.scheduler.config.num_train_timesteps, (fake_images.size(0),),
                                      device=config.device).long()

            # 将生成器的输出断开梯度连接
            fake_images_detached = fake_images.detach()

            # 扩散模型输出
            diffusion_output = diffusion_model(fake_images_detached, timesteps)

            # 假设扩散模型的目标是恢复真实图像
            diffusion_loss = nn.MSELoss()(diffusion_output, images)
            diffusion_loss.backward()
            optimizer_diffusion.step()

            # 打印损失
            if (i + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{config.num_epochs}], Step [{i + 1}/{len(dataloader)}], '
                    f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                    f'Diffusion Loss: {diffusion_loss.item():.4f}')

        # 保存生成的图像
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_images = images[:4]
                fake_images = generator(sample_images)
                # Generate timesteps
                timesteps = torch.randint(0, diffusion_model.scheduler.config.num_train_timesteps,
                                          (fake_images.size(0),),
                                          device=config.device).long()

                enhanced_images = diffusion_model(fake_images, timesteps)
                # 合并原图、生成图和增强图
                combined = torch.cat((sample_images, fake_images, enhanced_images), dim=0)
                save_image(combined, os.path.join(config.output_dir, f'epoch_{epoch + 1}.png'), nrow=4, normalize=True)

    # 保存模型
    torch.save(generator.state_dict(), os.path.join(config.output_dir, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(config.output_dir, 'discriminator.pth'))
    torch.save(diffusion_model.state_dict(), os.path.join(config.output_dir, 'diffusion_model.pth'))


if __name__ == "__main__":
    main()
