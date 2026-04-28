
import torch.nn as nn
from torchvision.ops import deform_conv2d
class Norm2d(nn.Module):
    """PyTorch 默认的 InstanceNorm2d"""

    def __init__(self, num_features, type='instanceNorm',eps=1e-5):
        super().__init__()
        if type == 'instanceNorm':
            self.norm = nn.InstanceNorm2d(num_features, eps=eps)

    def forward(self, x):
        return self.norm(x)


class CustomInstanceNormalization(nn.Module):
    """自定义实例归一化（根据需求实现）"""

    def __init__(self, second_component='batch'):
        super().__init__()
        # 这里需要根据你的 TensorFlow 实现来补充
        raise NotImplementedError("CustomInstanceNormalization 需要根据原始实现补充")


def upsample(in_channels, out_channels, kernel_size, norm_type='batchnorm', apply_dropout=False):
    """PyTorch 上采样块"""
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=kernel_size,
                                     stride=2,
                                     padding=1,
                                     bias=False))

    if norm_type.lower() == 'batchnorm':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm_type.lower() == 'instancenorm':
        layers.append(Norm2d(out_channels))
    elif norm_type.lower() == 'batchinstancenorm':
        layers.append(CustomInstanceNormalization(second_component='batch'))
    elif norm_type.lower() == 'layerinstancenorm':
        layers.append(CustomInstanceNormalization(second_component='layer'))

    if apply_dropout:
        layers.append(nn.Dropout2d(0.5))

    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def downsample(in_channels, out_channels, kernel_size, norm_type='batchnorm', apply_norm=True):
    """PyTorch 下采样块"""
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size,
                            stride=2,
                            padding=1,
                            bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm_type.lower() == 'instancenorm':
            layers.append(Norm2d(out_channels))
        elif norm_type.lower() == 'batchinstancenorm':
            layers.append(CustomInstanceNormalization(second_component='batch'))
        elif norm_type.lower() == 'layerinstancenorm':
            layers.append(CustomInstanceNormalization(second_component='layer'))

    layers.append(nn.LeakyReLU(0.2, inplace=True))

    return nn.Sequential(*layers)


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding

        # 主卷积层
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

        # 偏移量生成层（输出通道数为2 * kernel_size^2）
        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)

        # 初始化偏移量权重为0
        nn.init.normal_(self.offset_conv.weight,mean=0,std=0.01)
        # nn.init.constant_(self.offset_conv.weight, 0,std=0.01)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        # 生成偏移量
        offset = self.offset_conv(x)

        # 执行可变形卷积
        return deform_conv2d(x,
                             offset=offset,
                             weight=self.conv.weight,
                             bias=self.conv.bias,
                             stride=self.conv.stride,
                             padding=self.conv.padding)