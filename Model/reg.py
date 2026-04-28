# system
import os

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

# local
# from .layers import DownBlock, Conv, ResnetTransformer

from tools.layers import DownBlock,Conv,ResnetTransformer
sampling_align_corners = False

# The number of filters in each block of the encoding part (down-sampling).
ndf = {'A': [32, 64, 64, 64, 64, 64, 64], 'B':[8, 16, 16, 32, 32]}
# The number of filters in each block of the decoding part (up-sampling).
# If len(ndf[cfg]) > len(nuf[cfg]) - then the deformation field is up-sampled to match the input size.
nuf = {'A': [64, 64, 64, 64, 64, 64, 32],'B':[32, 32, 32, 32, 32, 16, 16] }
# Indicate if res-blocks are used in the down-sampling path.
use_down_resblocks = {'A': True, }
# indicate the number of res-blocks applied on the encoded features.
resnet_nblocks = {'A': 3, }
# Indicate if the a final refinement layer is applied on the before deriving the deformation field
refine_output = {'A': True, }
# The activation used in the down-sampling path.
down_activation = {'A': 'leaky_relu', }
# The activation used in the up-sampling path.
up_activation = {'A': 'leaky_relu', }


class ResUnet(torch.nn.Module):
    def __init__(self, nc_a, nc_b, cfg, init_func, init_to_identity):
        super(ResUnet, self).__init__()
        act = down_activation[cfg]
        # ------------ Down-sampling path
        self.ndown_blocks = len(ndf[cfg])
        self.nup_blocks = len(nuf[cfg])
        assert self.ndown_blocks >= self.nup_blocks
        in_nf = nc_a + nc_b
        conv_num = 1
        skip_nf = {}
        for out_nf in ndf[cfg]:
            setattr(self, 'down_{}'.format(conv_num),
                    DownBlock(in_nf, out_nf, 3, 1, 1, activation=act, init_func=init_func, bias=True,
                              use_resnet=use_down_resblocks[cfg], use_norm=False))
            skip_nf['down_{}'.format(conv_num)] = out_nf
            in_nf = out_nf
            conv_num += 1
        conv_num -= 1
        if use_down_resblocks[cfg]:
            self.c1 = Conv(in_nf, 2 * in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
            self.t = ((lambda x: x) if resnet_nblocks[cfg] == 0
                      else ResnetTransformer(2 * in_nf, resnet_nblocks[cfg], init_func))
            self.c2 = Conv(2 * in_nf, in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
        # ------------- Up-sampling path
        act = up_activation[cfg]
        for out_nf in nuf[cfg]:
            setattr(self, 'up_{}'.format(conv_num),
                    Conv(in_nf + skip_nf['down_{}'.format(conv_num)], out_nf, 3, 1, 1, bias=True, activation=act,
                         init_fun=init_func, use_norm=False, use_resnet=False))
            in_nf = out_nf
            conv_num -= 1
        if refine_output[cfg]:
            self.refine = nn.Sequential(ResnetTransformer(in_nf, 1, init_func),
                                        Conv(in_nf, in_nf, 1, 1, 0, use_resnet=False, init_func=init_func,
                                             activation=act,
                                             use_norm=False)
                                        )
        else:
            self.refine = lambda x: x
        ########## 在resnet backbone 后接上一个 卷积输出为2通道的特征图
        self.output = Conv(in_nf, 2, 3, 1, 1, use_resnet=False, bias=True,
                           init_func=('zeros' if init_to_identity else init_func), activation=None,
                           use_norm=False)

    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        skip_vals = {}
        conv_num = 1
        # Down
        while conv_num <= self.ndown_blocks:
            x, skip = getattr(self, 'down_{}'.format(conv_num))(x)
            skip_vals['down_{}'.format(conv_num)] = skip
            conv_num += 1
        if hasattr(self, 't'):
            x = self.c1(x)
            x = self.t(x)
            x = self.c2(x)
        # Up
        conv_num -= 1
        while conv_num > (self.ndown_blocks - self.nup_blocks):
            s = skip_vals['down_{}'.format(conv_num)]
            x = F.interpolate(x, (s.size(2), s.size(3)), mode='bilinear')
            x = torch.cat([x, s], 1)
            x = getattr(self, 'up_{}'.format(conv_num))(x)
            conv_num -= 1
        x = self.refine(x)
        x = self.output(x)
        return x


class Reg1(nn.Module):
    # backbone: resUnet
    def __init__(self,height,width,in_channels_a, in_channels_b):
        super(Reg1, self).__init__()
       #height,width=256,256
        #in_channels_a,in_channels_b=1,1
        init_func = 'kaiming'
        init_to_identity = True

        # paras end------------

        self.oh, self.ow = height, width
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offset_map = ResUnet(self.in_channels_a, self.in_channels_b, cfg='A', init_func=init_func, init_to_identity=init_to_identity).to(
            self.device)
        self.identity_grid = self.get_identity_grid()

    def get_identity_grid(self):
        x = torch.linspace(-1.0, 1.0, self.ow)
        y = torch.linspace(-1.0, 1.0, self.oh)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity

    def forward(self, img_a, img_b, apply_on=None):

        deformations = self.offset_map(img_a, img_b)

        return deformations


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ndims=2, stride=1):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            padding=1, stride=stride, bias=True
        )
        # self.conv = getattr(nn, f'Conv{ndims}d')(
        #     in_channels, out_channels, kernel_size=3,
        #     padding=1, stride=stride, bias=True
        # )
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        # He normal initialization  权重初始化
        nn.init.kaiming_normal_(self.conv.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.leaky_relu(self.conv(x))

# 使用的是尸检 中设计的vol R网络
class UNetCoreVJX(nn.Module):
    def __init__(self,enc_nf, dec_nf, src_feats=3, tgt_feats=3):
        super(UNetCoreVJX,self).__init__()

        self.ndims = 2  # 固定为2D
        # Encoder path
        self.encoder = nn.ModuleList()
        in_channels = src_feats + tgt_feats
        for nf in enc_nf:
            self.encoder.append(ConvBlock(in_channels, nf,stride=2))
            in_channels = nf
        # Bottleneck
        self.bottleneck = ConvBlock(enc_nf[-1], dec_nf[0], self.ndims)
        # Decoder path
        self.decoder = nn.ModuleList()
        for i in range(1, len(enc_nf) + 1):
            self.decoder.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            if i == len(enc_nf):
                skp_src = src_feats + tgt_feats
            else:
                skp_src = enc_nf[-(i + 1)]
            self.decoder.append(ConvBlock(dec_nf[i - 1] + skp_src, dec_nf[i], self.ndims))
            self.decoder.append(ConvBlock(dec_nf[i], dec_nf[i], self.ndims))
        # Final layers
        self.final_conv1 = ConvBlock(dec_nf[-2], dec_nf[-2], self.ndims)
        self.final_conv2 = ConvBlock(dec_nf[-2], dec_nf[-1], self.ndims)

        ########## 在resnet backbone 后接上一个 卷积输出为2通道的特征图
        self.output_flow = nn.Conv2d(dec_nf[-1],2,kernel_size=3,padding=1)
        nn.init.kaiming_normal_(self.output_flow.weight)

    def forward(self, src, tgt):
        # Input concatenation
        # batch,C ,H,W  通道拼接
        x = torch.cat([src, tgt], dim=1)

        # Encoder
        skip_connections = [x]
        for i in range(0, len(self.encoder)):
            x = self.encoder[i](x)  # ConvBlock
            skip_connections.append(x)
        # Bottleneck
        x = self.bottleneck(x)
        # Decoder
        skip_idx = len(skip_connections) - 2
        for i in range(0, len(self.decoder), 3):
            x = self.decoder[i](x)  # Upsample
            x = torch.cat([x, skip_connections[skip_idx]], dim=1)
            skip_idx -= 1
            x = self.decoder[i + 1](x)  # ConvBlock
            x = self.decoder[i + 2](x)  # ConvBlock
        # Final layers
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        # deformable field
        x = self.output_flow(x)
        return x


class Reg2(nn.Module):
    def __init__(self,height,width,in_channels_a, in_channels_b):
        super(Reg2, self).__init__()
       #height,width=256,256
        #in_channels_a,in_channels_b=1,1
        init_func = 'kaiming'
        init_to_identity = True

        # paras end------------
        self.nf_enc = [8, 16, 16, 32, 32]
        self.nf_dnc = [32, 32, 32, 32, 32, 16, 16]
        self.oh, self.ow = height, width
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offset_map = UNetCoreVJX(enc_nf=self.nf_enc,dec_nf=self.nf_dnc,src_feats=self.in_channels_a,
                                      tgt_feats=self.in_channels_b).to(self.device)

        self.identity_grid = self.get_identity_grid()

    def get_identity_grid(self):
        x = torch.linspace(-1.0, 1.0, self.ow)
        y = torch.linspace(-1.0, 1.0, self.oh)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity

    def forward(self, img_a, img_b, apply_on=None):

        deformations = self.offset_map(img_a, img_b)

        return deformations


# 使用 Attention Unet 作为R网络
class Attention_block(nn.Module):
    # 注意力块
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


# 这里时注意力机制
class AttU_Net(nn.Module):
    def __init__(self,src_feature=3,tgt_feature=3 ,scale_factor=1):
        super(AttU_Net, self).__init__()
        # filters = [64, 128, 256, 512, 1024]
        filters = [16, 32, 64, 128, 256]
        # filters = [16, 32, 32, 64, 64,]
        filters = [(f // scale_factor) for f in filters]
        # filters = filters // scale_factor
        self.n_channels = src_feature + tgt_feature
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder = nn.ModuleList()
        # tmp = self.n_channels
        # for f in filters:
        #     self.encoder.append(conv_block(ch_in=tmp,ch_out=f))
        #     tmp = f
        self.Conv1 = conv_block(ch_in=self.n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])
        # self.decoder = nn.ModuleList()
        # for i in range(len(filters)-1,1,-1):
        #     self.decoder.append(up_conv(ch_in=filters[i],ch_out=filters[i-1]))
        #     self.decoder.append(Attention_block(F_g=filters[i-1], F_l=filters[i-1], F_int=filters[i-2]))
        #     self.decoder.append(conv_block(ch_in=filters[i], ch_out=filters[i-1]))
        # self.decoder.append(up_conv(ch_in=filters[1], ch_out=filters[0]))
        # self.decoder.append(Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2))
        # self.decoder.append(conv_block(ch_in=filters[1], ch_out=filters[0]))
        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])

        self.output_flow = nn.Conv2d(filters[0],2,kernel_size=3,padding=1)
        nn.init.kaiming_normal_(self.output_flow.weight)


    def forward(self, src,tgt):
        x = torch.cat([src, tgt], dim=1)
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

        fm = self.output_flow(d2)

        return fm


class Reg3(nn.Module):
    def __init__(self,height,width,in_channels_a, in_channels_b):
        super(Reg3, self).__init__()
       #height,width=256,256
        #in_channels_a,in_channels_b=1,1
        init_func = 'kaiming'
        init_to_identity = True
        # paras end------------
        # self.nf_enc = [8, 16, 16, 32, 32]
        # self.nf_dnc = [32, 32, 32, 32, 32, 16, 16]
        self.oh, self.ow = height, width
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offset_map = AttU_Net(src_feature=self.in_channels_a,tgt_feature=self.in_channels_b).to(self.device)

    #     self.identity_grid = self.get_identity_grid()
    #
    # def get_identity_grid(self):
    #     x = torch.linspace(-1.0, 1.0, self.ow)
    #     y = torch.linspace(-1.0, 1.0, self.oh)
    #     xx, yy = torch.meshgrid([y, x])
    #     xx = xx.unsqueeze(dim=0)
    #     yy = yy.unsqueeze(dim=0)
    #     identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
    #     return identity

    def forward(self, img_a, img_b, apply_on=None):
        deformations = self.offset_map(img_a, img_b)
        return deformations


if __name__ == '__main__':
    src_img = torch.randn(1,3,  256, 256)
    tat_img = torch.randn(1, 3, 256, 256)
    attu = AttU_Net(3,3)
    print(attu(src_img,tat_img).shape)
