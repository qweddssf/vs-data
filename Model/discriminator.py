import torch
import torch.nn as nn
import torch.nn.functional as F
from .structure import downsample,CustomInstanceNormalization,Norm2d,DeformableConv2d

class PatchGANDiscriminator(nn.Module):
    """PyTorch PatchGAN判别器"""

    def __init__(self, in_channels=3, norm_type='batchnorm', target=True):
        super().__init__()
        self.target = target

        # 如果使用目标图像，输入通道数加倍
        input_channels = in_channels * 2 if target else in_channels

        self.model = nn.Sequential(
            # 第一层不使用归一化
            *downsample(input_channels, 64, 4, norm_type, apply_norm=False),  # [bs, 64, 128, 128]
            *downsample(64, 128, 4, norm_type),  # [bs, 128, 64, 64]
            *downsample(128, 256, 4, norm_type),  # [bs, 256, 32, 32]

            # 零填充+卷积块1
            nn.ZeroPad2d(1),  # [bs, 256, 34, 34]
            nn.Conv2d(256, 512, kernel_size=4, stride=1, bias=False),  # [bs, 512, 31, 31]
            self._get_norm_layer(512, norm_type),  # 归一化层
            nn.LeakyReLU(0.2, inplace=True),

            # 零填充+最后卷积
            nn.ZeroPad2d(1),  # [bs, 512, 33, 33]
            nn.Conv2d(512, 1, kernel_size=4, stride=1),  # [bs, 1, 30, 30]
            nn.Sigmoid()
        )

    def _get_norm_layer(self, channels, norm_type):
        """创建归一化层"""
        if norm_type.lower() == 'batchnorm':
            return nn.BatchNorm2d(channels)
        elif norm_type.lower() == 'instancenorm':
            return Norm2d(channels)
        elif norm_type.lower() == 'batchinstancenorm':
            return CustomInstanceNormalization(second_component='batch')
        elif norm_type.lower() == 'layerinstancenorm':
            return CustomInstanceNormalization(second_component='layer')
        else:
            raise ValueError(f"未知的归一化类型: {norm_type}")

    def forward(self, input_image, target_image=None):
        if self.target:
            # 验证目标图像是否提供
            if target_image is None:
                raise ValueError("模型配置需要目标图像，但未提供target_image")
            # 在通道维度拼接输入和目标图像
            x = torch.cat([input_image, target_image], dim=1)  # [bs, 6, 256, 256]
        else:
            x = input_image

        return self.model(x)

############################
# 基础的 判别器
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten

        return F.sigmoid(F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1))


############################
class Discriminator_withDeconv(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator_withDeconv,self).__init__()

        model = [
            # nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 DeformableConv2d(input_nc,64,4,stride=2,padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [
            # nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  DeformableConv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [DeformableConv2d(128,256,4,2,1),
                  # nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [DeformableConv2d(256,512,4,padding=1),
                  # nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


############################
class Discriminator_att(nn.Module):
    def __init__(self, input_nc=3,type='batchNorm',scale=1):
        super(Discriminator_att, self).__init__()

        # A bunch of convolutions one after another
        filter = [32,64,64,128,128,256,256,512,512]
        filter = [ f // scale for f in filter]
        # filter = [16, 32, 32, 64, 64, 128, 128, 256, 256]  # 目前最好的状态
        models = [nn.Conv2d(input_nc, filter[0], 3, stride=1, padding=1),
                # selectNorm(num_features=filter[0], type=type),
                 nn.LeakyReLU(0.2, inplace=True)]

        for i in range(1,len(filter),2):
            models += [nn.Conv2d(filter[i-1], filter[i], 3, stride=2, padding=1),
                 # selectNorm(num_features=filter[i], type=type),
                 nn.LeakyReLU(0.2, inplace=True)]
            models += [nn.Conv2d(filter[i], filter[i], 3, stride=1, padding=1),
                 # selectNorm(num_features=filter[i], type=type),
                 nn.LeakyReLU(0.2, inplace=True)]

        self.models = nn.Sequential(*models)
        self.Fc1 = nn.Linear(filter[-1], filter[-1])
        self.LRelu = nn.LeakyReLU()
        self.Fc2 = nn.Linear(filter[-1],1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.models(x)
        x_mean = torch.mean(x,dim=(2,3))
        x =self.LRelu(self.Fc1(x_mean))
        x = self.Fc2(x)
        x = self.sigmoid(x)
        return x
        # Average pooling and flatten
        # return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


############################
class PatchGANDiscriminator2(nn.Module):
    def __init__(self, input_channels=3, use_sigmoid=False):
        """
        PatchGAN 判别器
        - input_channels: 输入图像的通道数（例如，RGB 为 3，条件GAN中可能为 6）
        - use_sigmoid: 是否在最后一层使用 Sigmoid（标准GAN用，LSGAN不需要）
        """
        super().__init__()
        self.model = nn.Sequential(
            # 输入尺寸: (input_channels) x 256 x 256
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 128 x 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x 64 x 64
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 x 32 x 32
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 512 x 16 x 16 → 输出 1 x 14 x 14（每个点对应一个图像块）
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Sigmoid() if use_sigmoid else nn.Identity()  # 根据损失函数选择
        )

    def forward(self, x):
        return self.model(x)
