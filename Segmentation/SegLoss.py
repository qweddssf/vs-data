import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
def dice_coef(y_true,y_pred):
    smooth = 1e-6
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
            torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce_loss = F.binary_cross_entropy(y_pred, y_true)
    return 0.5 * bce_loss + dice_loss(y_true, y_pred)


# 这个loss 用来约束细胞核数量
def soft_count_loss(vs_prob, gt_prob):
    """
    基于概率密度总和的软数量约束
    vs_prob: [B,C, H, W] (经过Sigmoid)
    gt_prob: [B,C, H, W] (经过Sigmoid)
    """
    # 计算每张图像的细胞概率总和（可导）
    # count_vs = vs_prob.sum(dim=(1,2))  # [B]
    # count_gt = gt_prob.sum(dim=(1,2))  # [B]

    # 平均密度差异
    # count_vs = vs_prob.mean(dim=(1,2))  # [B]
    # count_gt = gt_prob.mean(dim=(1,2))  # [B]
    # # 使用L1损失约束总量
    # return F.l1_loss(count_vs, count_gt)

    # 使用MSE损失约束总量
    # 平均密度差异
    # count_vs = vs_prob.sum(dim=(1,2,3))  # [B,c,h,w]
    # count_gt = gt_prob.sum(dim=(1,2,3))  # [B,c,h,w]
    # F.mse_loss(count_vs, count_gt, reduction='mean') * 0.001
    bce_loss = F.binary_cross_entropy(vs_prob,gt_prob,reduction='mean')
    return  bce_loss


# class CountFocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, vs_prob, gt_mask):
#         # 计算数量差异权重
#         count_vs = vs_prob.sum(dim=(1, 2, 3))
#         count_gt = gt_mask.sum(dim=(1, 2, 3))
#         weight = 1 + torch.abs(count_vs - count_gt)  # 数量差异越大，惩罚越重
#
#         # Focal Loss计算
#         bce = F.binary_cross_entropy(vs_prob, gt_mask, reduction='none')
#         pt = torch.exp(-bce)
#         focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce).mean(dim=(1, 2, 3))
#
#         return (weight * focal_loss).mean()

def CountFocalLoss(vs_prob,gt_mask, alpha=0.25, gamma=2):
    # 计算数量差异权重
    count_vs = vs_prob.sum(dim=(1, 2, 3))
    count_gt = gt_mask.sum(dim=(1, 2, 3))
    weight = 1 + torch.abs(count_vs - count_gt)  # 数量差异越大，惩罚越重
    bce = F.binary_cross_entropy(vs_prob, gt_mask, reduction='none')
    pt = torch.exp(-bce)
    focal_loss = (alpha * (1 - pt) ** gamma * bce).mean(dim=(1, 2, 3))

    return (weight * focal_loss).mean()




# 自适应 阈值得到 mask
def adaptive_threshold(cell_prob, alpha=4.0):
    """基于概率图均值的动态阈值"""
    mean_prob = cell_prob.mean()
    threshold = alpha * mean_prob
    binary_mask = (cell_prob > threshold).float()
    return cell_prob + (binary_mask - cell_prob).detach()

# 在细胞核区域加强对抗约束，背景区域放宽
# 方法：使用分割掩码加权对抗损失
# 效果：在细胞核区域追求高保真，背景允许更多多样性。
def masked_adv_loss(d_real, d_fake, gt_mask):
    """在细胞核区域施加更强对抗约束"""
    # 生成掩码权重（细胞核区域权重=2，背景=0.5）
    weights = gt_mask * 2.0 + (1 - gt_mask) * 0.5

    # 加权对抗损失
    loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real), weight=weights)
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake), weight=weights)
    return (loss_real + loss_fake) * 0.5


class AdaptiveLossWeight:
    """改进版动态权重调整器 针对直接显示计算细胞核数量的 权重平衡问题"""

    def __init__(self, target_ratio=0.5, init_weight=1e-5, ema_alpha=0.9, max_weight=0.0001, min_weight=1e-6):
        self.target_ratio = target_ratio
        self.current_weight = init_weight
        self.ema_alpha = ema_alpha
        self.loss_ratio_ema = None
        self.max_weight = max_weight
        self.min_weight = min_weight

    def update(self, loss_focal, loss_adv):
        current_ratio = loss_focal.item() / (loss_adv.item() + 1e-8)

        # 更新EMA损失比例
        if self.loss_ratio_ema is None:
            self.loss_ratio_ema = current_ratio
        else:
            self.loss_ratio_ema = self.ema_alpha * self.loss_ratio_ema + (1 - self.ema_alpha) * current_ratio

        # 计算调整因子（关键改进）
        adjustment = (self.loss_ratio_ema / self.target_ratio) ** 0.1

        # 应用调整并约束范围
        self.current_weight *= adjustment
        self.current_weight = max(min(self.current_weight, self.max_weight), self.min_weight)

        return self.current_weight

# 初始化权重管理器
weight_manager = AdaptiveLossWeight(init_weight=5e-5)
# 分割一致loss
def segmentation_loss(vs_seg, gt_seg,weight_dP=0.5, weight_dice=1.0,weight_cnt = 1.0):
    """计算生成图像与真实图像在分割特征空间的差异"""
    # 分解特征
    vs_dP, vs_logits = vs_seg
    gt_dP, gt_logits = gt_seg

    # b,W,H ---> b,1,W,H
    vs_prob = torch.sigmoid(vs_logits).unsqueeze(1)
    gt_prob = torch.sigmoid(gt_logits.detach()).unsqueeze(1)  # 阻止梯度流向gt

    # 约束细胞核数量的损失 隐式约束细胞核数量
    loss_cnt = soft_count_loss(vs_prob, gt_prob)
    # 显式控制
    # loss_cnt = F.l1_loss(vs_prob,gt_prob)

    # 分割的概率密度图的     相似性损失1
    loss_dice = dice_loss(gt_prob,vs_prob)

    # loss_dice_bce = bce_dice_loss(gt_prob,vs_prob)
    # 不用dice loss 防止对于细胞位置的分布约束梯度 和adv 的多样性冲突

    # loss_cnt2 = CountFocalLoss(vs_prob, gt_mask)
    # gt_mask = adaptive_threshold(gt_prob)
    # vs_mask = adaptive_threshold(vs_prob)
    # plt.imsave('./test_img/Seg_vision_dir/gt_mask.png',gt_mask.squeeze(dim=1).cpu().numpy().squeeze())
    # plt.imsave('./test_img/Seg_vision_dir/vs_mask.png', vs_mask.squeeze(dim=1).cpu().numpy().squeeze())
    # plt.imsave('./test_img/Seg_vision_dir/sample12_gt.png',gt_prob.squeeze(dim=1).cpu().numpy().squeeze())
    # plt.imsave('./test_img/Seg_vision_dir/sample12_vs.png', vs_prob.squeeze(dim=1).cpu().numpy().squeeze())


    # 这里刚开始的loss 似乎太大了 需要 权重调整为 0.00005

    # focal_loss = CountFocalLoss(vs_prob=vs_prob,gt_mask=gt_mask)

    # 流场梯度差异（方向敏感）相似性损失2 这里似乎可以隐式控制数量
    # loss_dP = 1 - torch.cosine_similarity(vs_dP, gt_dP, dim=1).mean()

    # loss_dP = loss_dP + 0.1 * ( (vs_dP.norm(dim=1) - gt_dP.norm(dim=1)).abs().mean() )

    return loss_dice * weight_dice, loss_cnt * weight_cnt
