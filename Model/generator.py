import torch
import torch.nn as nn

from .structure import upsample, downsample,DeformableConv2d
##############################################
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out,type):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            selectNorm(num_features=ch_out, type=type),
            # nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            selectNorm(num_features=ch_out, type=type),
            # nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, type):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            selectNorm(num_features=ch_out,type=type),
            # nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class Attention_block(nn.Module):
    # 注意力块
    def __init__(self, F_g, F_l, F_int,type):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            selectNorm(num_features=F_int, type=type),
            # nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            selectNorm(num_features=F_int, type=type),
            # nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            selectNorm(num_features=1,type=type),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

def selectNorm( num_features, type, eps=1e-5):
    if type == 'batchNorm':
        return nn.BatchNorm2d(num_features,eps)
    elif type == 'instanceNorm':
        return nn.InstanceNorm2d(num_features,eps)


class Gan_AttU_Net(nn.Module):
    def __init__(self,inp_feature=3 ,out_feature=3, scale_factor=2,type='batchNorm'):
        super(Gan_AttU_Net, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        # filters = [32, 64, 128, 256, 512] # exp25使用
        filters = [(f // scale_factor) for f in filters]
        self.n_channels = inp_feature
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=self.n_channels, ch_out=filters[0],type=type)
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1],type=type)
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2],type=type)
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3],type=type)
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4],type=type)

        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3],type=type)
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2],type=type)
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3],type=type)

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2],type=type)
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1],type=type)
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2],type=type)

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1],type=type)
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0],type=type)
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1],type=type)

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0],type=type)
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2,type=type)
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0],type=type)

        self.Conv = nn.Conv2d(filters[0], out_feature, kernel_size=1, stride=1, padding=0)
        # nn.init.xavier_normal(self.Conv.weight)
        nn.init.xavier_normal_(self.Conv.weight)
        self.Sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax()

    def forward(self, x):

        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)

        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        # output = self.Conv(d2)
        output = self.Sigmoid(self.Conv(d2))
        return output



############################################## pix2pix gan 的一种结构变化（一些卷积层的变化）
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


class UNetGenerator(nn.Module):
    """PyTorch U-Net 生成器"""

    def __init__(self, in_channels=3, out_channels=3, norm_type='batchnorm'):
        super().__init__()

        # 下采样路径 (编码器)
        self.down1 = downsample(in_channels, 64, 4, norm_type, apply_norm=False)
        self.down2 = downsample(64, 128, 4, norm_type)
        self.down3 = downsample(128, 256, 4, norm_type)
        self.down4 = downsample(256, 512, 4, norm_type)

        self.down5 = downsample(512, 512, 4, norm_type)
        self.down6 = downsample(512, 512, 4, norm_type)
        self.down7 = downsample(512, 512, 4, norm_type)
        self.down8 = downsample(512, 512, 4, norm_type)

        # 上采样路径 (解码器)
        self.up1 = upsample(512, 512, 4, norm_type, apply_dropout=True)
        self.up2 = upsample(1024, 512, 4, norm_type, apply_dropout=True)
        self.up3 = upsample(1024, 512, 4, norm_type, apply_dropout=True)
        self.up4 = upsample(1024, 512, 4, norm_type)
        self.up5 = upsample(1024, 256, 4, norm_type)
        self.up6 = upsample(512, 128, 4, norm_type)
        self.up7 = upsample(256, 64, 4, norm_type)

        # 最终输出层
        self.last = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, x):
        # 下采样
        d1 = self.down1(x)  # [bs, 64, 128, 128]
        d2 = self.down2(d1)  # [bs, 128, 64, 64]
        d3 = self.down3(d2)  # [bs, 256, 32, 32]
        d4 = self.down4(d3)  # [bs, 512, 16, 16]
        d5 = self.down5(d4)  # [bs, 512, 8, 8]
        d6 = self.down6(d5)  # [bs, 512, 4, 4]
        d7 = self.down7(d6)  # [bs, 512, 2, 2]
        d8 = self.down8(d7)  # [bs, 512, 1, 1]

        # 上采样并连接跳跃连接
        u1 = self.up1(d8)  # [bs, 512, 2, 2]
        u1 = torch.cat([u1, d7], dim=1)  # [bs, 1024, 2, 2]

        u2 = self.up2(u1)  # [bs, 512, 4, 4]
        u2 = torch.cat([u2, d6], dim=1)  # [bs, 1024, 4, 4]

        u3 = self.up3(u2)  # [bs, 512, 8, 8]
        u3 = torch.cat([u3, d5], dim=1)  # [bs, 1024, 8, 8]

        u4 = self.up4(u3)  # [bs, 512, 16, 16]
        u4 = torch.cat([u4, d4], dim=1)  # [bs, 1024, 16, 16]

        u5 = self.up5(u4)  # [bs, 256, 32, 32]
        u5 = torch.cat([u5, d3], dim=1)  # [bs, 512, 32, 32]

        u6 = self.up6(u5)  # [bs, 128, 64, 64]
        u6 = torch.cat([u6, d2], dim=1)  # [bs, 256, 64, 64]

        u7 = self.up7(u6)  # [bs, 64, 128, 128]
        u7 = torch.cat([u7, d1], dim=1)  # [bs, 128, 128, 128]

        return self.last(u7)  # [bs, out_channels, 256, 256]


############################################# 可行变卷积 或  double conv


class ResidualBlock_deconv(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock_deconv, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            DeformableConv2d(in_channels=in_features,out_channels=in_features,kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            DeformableConv2d(in_channels=in_features, out_channels=in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features)]
        self.deconv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.deconv_block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator_with_decov(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super( Generator_with_decov, self).__init__()

        # Initial convolution block
        model_head = [nn.ReflectionPad2d(3),
                      DeformableConv2d(input_nc,64,7),
                      # nn.Conv2d(input_nc, 64, 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [DeformableConv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(in_features)]
            # model_body += [ResidualBlock_deconv(in_features)]
        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [nn.ReflectionPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)


    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)
        return x

############################################