import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """修正后的注意力门机制"""

    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

        # 添加初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, g):
        # 处理门控信号
        g_conv = self.W_g(g)
        # 处理跳跃连接特征
        x_conv = self.W_x(x)

        # 调整尺寸
        if g_conv.shape[2:] != x_conv.shape[2:]:
            g_conv = F.interpolate(g_conv, size=x_conv.shape[2:], mode='bilinear', align_corners=True)

        # 融合特征
        fused = self.relu(g_conv + x_conv)
        # 生成注意力权重
        attention = self.psi(fused)
        # 应用注意力权重
        return x * attention


class DoubleConv(nn.Module):
    """双重卷积模块（卷积+BN+ReLU）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 添加初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """下采样模块（包含最大池化和双重卷积）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """上采样模块（包含上采样和双重卷积）"""

    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # 添加注意力门
        self.use_attention = use_attention
        if use_attention:
            self.attention_gate = AttentionGate(in_channels // 2, in_channels // 2, in_channels // 4)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 上采样
        x1 = self.up(x1)

        # 处理特征图尺寸差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 应用注意力门
        if self.use_attention:
            x2 = self.attention_gate(x2, x1)

        # 拼接特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Generator(nn.Module):
    """基于Attention U-Net的生成器"""

    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        # 初始下采样
        self.inc = DoubleConv(in_channels, features)
        self.down1 = DownBlock(features, features * 2)
        self.down2 = DownBlock(features * 2, features * 4)
        self.down3 = DownBlock(features * 4, features * 8)
        self.down4 = DownBlock(features * 8, features * 16)

        # 上采样路径（带注意力门）
        self.up1 = UpBlock(features * 16, features * 8, use_attention=True)
        self.up2 = UpBlock(features * 8, features * 4, use_attention=True)
        self.up3 = UpBlock(features * 4, features * 2, use_attention=True)
        self.up4 = UpBlock(features * 2, features, use_attention=False)

        # 输出层
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

        # 初始化输出层
        nn.init.kaiming_normal_(self.outc.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码路径（带跳跃连接）
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出层
        return torch.tanh(self.outc(x))



class Discriminator(nn.Module):
    """改进的PatchGAN判别器"""

    def __init__(self, in_channels=3, features=64, num_layers=3):
        super().__init__()
        layers = []

        # 输入层
        layers.append(nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 中间层
        for i in range(1, num_layers):
            in_chs = features * min(2 ** (i - 1), 8)
            out_chs = features * min(2 ** i, 8)
            layers.extend([
                nn.Conv2d(in_chs, out_chs, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_chs),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        # 输出层
        layers.append(nn.Conv2d(features * min(2 ** (num_layers - 1), 8), 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """安全的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight'):
                    nn.init.normal_(m.weight, 0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


# 测试代码
if __name__ == "__main__":
    # 超参数
    in_channels = 3
    img_size = 256
    batch_size = 2

    # 创建模型
    generator = Generator(in_channels, in_channels)
    discriminator = Discriminator(in_channels * 2)  # 输入为真实图像和生成图像的拼接

    # 测试生成器
    input_img = torch.randn(batch_size, in_channels, img_size, img_size)
    generated_img = generator(input_img)
    print(f"Generator output shape: {generated_img.shape}")  # 应为 [batch, 3, 256, 256]

    # 测试判别器 (输入真实图像和生成图像的拼接)
    real_img = torch.randn(batch_size, in_channels, img_size, img_size)
    disc_input = torch.cat([real_img, generated_img], dim=1)
    disc_output = discriminator(disc_input)
    print(f"Discriminator output shape: {disc_output.shape}")  # 应为 [batch, 1, h, w]