import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

# NCC loss
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def ncc(self, Ii, Ji):
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size,nb_feats,W,H]
        # ndims = Ii.dim() - 2  # excluding batch_size and channel dimensions
        ndims = len(Ii.shape) - 2
        in_ch = Ji.shape[1]
        assert ndims in [1, 2, 3], f"Volumes should be 1 to 3 dimensions. Found: {ndims}"
        # set window size0
        if self.win is None:
            self.win = [9] * ndims
        elif not isinstance(self.win, list):  # user specified a single number, not a list
            self.win = [self.win] * ndims

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # compute filters (all ones filter of the same shape as the window)
        sum_filt = torch.ones([in_ch] + self.win, device=Ii.device)
        sum_filt = sum_filt.unsqueeze(0)  # Add batch and channel dimensions shape out_c,in_c,H,W

        # compute local sums via convolution
        if ndims == 2:
            # 计算2D卷积的padding，确保输出尺寸与输入相同
            # 假设win是[h, w]，计算每个维度的padding
            pad_h = (self.win[0] - 1) // 2
            pad_w = (self.win[1] - 1) // 2
            padding = (pad_h, pad_w)  # 注意PyTorch的padding顺序是(上下, 左右)
        else:
            padding = 'valid'

        # Ii shape = (batch,in_ch, H,W)  sum_fit shape = (out_c, in_c,H,W)
        I_sum = F.conv2d(Ii, sum_filt, stride=1, padding=padding)
        J_sum = F.conv2d(Ji, sum_filt, stride=1, padding=padding)
        I2_sum = F.conv2d(I2, sum_filt, stride=1, padding=padding)
        J2_sum = F.conv2d(J2, sum_filt, stride=1, padding=padding)
        IJ_sum = F.conv2d(IJ, sum_filt, stride=1, padding=padding)

        # compute cross correlation
        win_size = torch.prod(torch.tensor(self.win, device=Ii.device)).item() * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = torch.maximum(cross, torch.tensor(self.eps))

        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = torch.maximum(I_var, torch.tensor(self.eps))

        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = torch.maximum(J_var, torch.tensor(self.eps))

        # NCC = (cross / I_var) * (cross / J_var)
        cc = (cross / I_var) * (cross / J_var)

        # return mean cc for each entry in batch
        return cc.view(cc.size(0), -1).mean(dim=-1).mean()
        # return torch.mean(cc.view(cc.size(0), -1),dim=-1)
    def loss(self, y_true, y_pred):
        return 1 - self.ncc(y_true, y_pred)

# bce loss
def bce_loss():
    return torch.nn.BCELoss()

# mse loss
def mse_loss():
    return torch.nn.MSELoss()

# l1 loss
def n1_loss():
    return torch.nn.L1Loss()

# dvf loss (grad loss)
def smooothing_loss(y_pred):
    # grad loss
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    dx = dx*dx
    dy = dy*dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d
    return d


class GradientPenaltyLoss(nn.Module):
    def __init__(self, lambda_gp=10):
        super(GradientPenaltyLoss, self).__init__()
        self.lambda_gp = lambda_gp

    def forward(self, discriminator, real_data, fake_data):
        """
        计算梯度惩罚损失
        Args:
            discriminator: 判别器网络
            real_data: 真实数据
            fake_data: 生成数据
        """
        batch_size = real_data.size(0)
        device = real_data.device

        # 在真实数据和生成数据之间随机插值
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        # 扩展alpha到与real_data相同的维度
        alpha = alpha.expand_as(real_data)

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_(True)

        # 计算判别器对插值点的输出
        disc_interpolates = discriminator(interpolates)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 计算梯度范数并计算惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return self.lambda_gp * gradient_penalty

# if __name__ == '__main__':
#     vs_img = np.array(plt.imread('vs.jpg').astype(np.float32)) / 255.0
#     gt_img = np.array(plt.imread('gt.jpg').astype(np.float32)) / 255.0
#     vs_img = torch.tensor(np.expand_dims(vs_img,axis=0))
#     gt_img = torch.tensor(np.expand_dims(gt_img,axis=0))
#
#     ncc_loss = NCC().loss(gt_img,vs_img)
#     print(ncc_loss)
#     sl = smooothing_loss(gt_img)
#     print(sl)
#     pass
