import torch
import torch.nn as nn
from cellpose import models
import matplotlib.pyplot as plt
import torch.nn.functional as F

sampling_align_corners = False
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
    def __init__(self, n_channels=3, n_classes=1, scale_factor=2):
        super(AttU_Net, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [(f // scale_factor) for f in filters]

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])

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

        self.Conv_1x1 = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
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

        d1 = self.Conv_1x1(d2)

        return self.sigmoid(d1)


class CellposeSegWrapper(nn.Module):
    """将Cellpose封装为可嵌入的模块，隔离梯度"""

    def __init__(self, model_type='cyto'):
        super(CellposeSegWrapper,self).__init__()
        self.cellpose_net = models.CellposeModel(gpu=True, model_type=model_type).net
        # 提取底层 PyTorch 网络
        # self.net = self.cellpose.net  # 实际承载参数的 PyTorch 模型

        # 冻结所有参数（核心步骤）
        for param in self.cellpose_net.parameters():
            param.requires_grad = False

        # 设置网络为评估模式（影响 BatchNorm/Dropout 等层）
        self.cellpose_net.eval()

    def preprocess(self, x):
        """将三通道RGB输入转换为Cellpose需要的双通道格式"""
        # 转换为灰度图 [B, 1, H, W]
        x_gray = x.mean(dim=1, keepdim=True)

        # 添加全零直径通道 [B, 2, H, W]
        dummy_diam = torch.zeros_like(x_gray)
        return torch.cat([x_gray, dummy_diam], dim=1)

    def forward(self, x):
        # 预处理
        x_in = self.preprocess(x)

        # 前向传播（直接调用底层网络）
        with torch.no_grad():  # 确保Cellpose不参与梯度计算
            y, _, _ = self.cellpose_net(x_in)

        # 提取关键特征
        dP = y[:, :2]  # 流场梯度 [B, 2, H, W]
        cellprob = y[:, 2]  # 细胞概率图 [B, H, W]
        return dP, cellprob

class DifferentiableConnectedComponents(nn.Module):
    """可导连通区域标记近似"""
    def __init__(self, kernel_size=5,cell_diameter=30):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.MaxPool_ker = cell_diameter // 2 * 2 + 1
        # nn.init.eye_(self.conv.weight)  # 初始化卷积核为恒等映射
        nn.init.dirac_(self.conv.weight)

    def forward(self, binary_mask,str):
        # 利用卷积模拟区域合并

        merged = torch.sigmoid(self.conv(binary_mask))
        # 检测局部极大值作为独立区域
        max_pool = nn.MaxPool2d(kernel_size=self.MaxPool_ker, stride=1, padding=self.MaxPool_ker//2)
        local_max = (merged == max_pool(merged)).float()
        plt.imsave(f'{str}.jpg',local_max.cpu().squeeze().numpy())
        # return local_max.sum(dim=(1,2))
        return local_max

def denoise_mask(binary_mask, iterations=1):
    """可导的形态学开运算（先腐蚀后膨胀）"""
    kernel_size = 5
    eroded = -F.max_pool2d(-binary_mask, kernel_size , stride=1, padding=kernel_size // 2)  # 腐蚀
    opened = F.max_pool2d(eroded, kernel_size , stride=1, padding= kernel_size // 2)         # 膨胀
    return opened

def f(prob,str):
    diff_cc = DifferentiableConnectedComponents().cuda()
    bm = (prob > 0.7).float()
    bm = denoise_mask(bm)
    local_max = diff_cc(bm,str)
    local_max = differentiable_nms(local_max,prob)
    #
    plt.imsave(f'{str}2.jpg', local_max.cpu().squeeze().numpy())
    return local_max.sum(dim=(1,2))


def differentiable_nms(local_max, cellprob, window=37):
    """抑制窗口内的次大值点"""
    # Step 1: 找到局部最大值
    max_val = F.max_pool2d(cellprob, window, stride=1, padding=window // 2)

    # Step 2: 仅保留严格最大值点（避免平局）
    is_max = (cellprob == max_val).float()

    # Step 3: 与原极大值图取交集
    return local_max * is_max

def asymmetric_count_loss(count_vs, count_gt, alpha=2.0):
    """非对称损失（少生成时惩罚更大）"""
    import torch.nn.functional as F
    diff = count_gt - count_vs
    loss_under = F.relu(diff).mean() * alpha  # 少生成时损失放大
    loss_over = F.relu(-diff).mean()          # 多生成时正常惩罚
    return loss_under + loss_over


if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()

    tissue_name = 'skin_tissue'
    patch_name = 'sample_gt_14_9'

    vs_path = f'test_img/{tissue_name}/vs_stained/{patch_name}'.replace('gt','vs')
    gt_path = f'test_img/{tissue_name}/gt_stained/{patch_name}'
    image_vs1 = Image.open(vs_path).convert("RGB")
    image_gt1 = Image.open(gt_path).convert("RGB")


    img_vs_tensor = transform(image_vs1).unsqueeze(0).cuda()
    img_gt_tensor = transform(image_gt1).unsqueeze(0).cuda()

    cel_model = CellposeSegWrapper()

    vs_dP, vs_logits =  cel_model(img_vs_tensor)
    gt_dP, gt_logits = cel_model(img_gt_tensor)
    vs_prob = torch.sigmoid(vs_logits).unsqueeze(1)
    gt_prob = torch.sigmoid(gt_logits).unsqueeze(1)
    plt.imsave('test_img/prostate_tissue/Seg_vision_dir/sample46_gt_prob_cell.png', gt_prob.squeeze(dim=1).cpu().numpy().squeeze())
    plt.imsave('test_img/prostate_tissue/Seg_vision_dir/sample46_vs_prob_cell.png', vs_prob.squeeze(dim=1).cpu().numpy().squeeze())



    # from Segmentation.SegLoss import segmentation_loss
    # vs_prob, gt_prob = segmentation_loss(vs_seg,gt_seg)
    # print('vs:',vs_prob,'gt:',gt_prob)


    # vs_num = f(vs_prob,'vs')
    # gt_num = f(gt_prob,'gt')
    # print('vs:',vs_num,'gt:',gt_num)
    #
    # print(F.l1_loss(vs_num,gt_num))
    # loss = asymmetric_count_loss(vs_num,gt_num, alpha=2.0)
    # print('loss',loss)
    # print(F.l1_loss(vs_prob,gt_prob))
    # dp, cellprob = cel_model(img_vs_tensor)
    # dp1, cellprob1 = cel_model(img_vs_tensor)


