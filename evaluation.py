import os
import numpy as np
import cv2
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt



def PSNR(fake, real):
    mse = np.mean((fake - real) ** 2)
    if mse < 1e-10:
        return 100
    MAX_pixel = 1.0;
    psnr = 20 * np.log10(MAX_pixel / np.sqrt(mse))
    return psnr


def MAE(fake, real):
    if real.ndim != 3:
        raise ValueError("real array is not 3-dimensional")

        # 初始化 MAE 列表
    maes = []

    # 遍历每个通道
    for channel in range(real.shape[0]):  # 假设通道是第一个维度
        x, y = np.where(real[channel] != -1)  # 选择非背景
        mae = np.abs(fake[channel][x, y] - real[channel][x, y]).mean()  # 计算当前通道的 MAE
        maes.append(mae)

    # 返回所有通道 MAE 的平均值，并归一化
    return np.mean(maes) / 2


def SSIM(fake,real):
    mssim, _, _ = compare_ssim(real, fake, channel_axis=0, gradient=True, data_range=1, full=True)
    return mssim


##### Segmentation eva
def cal_pcc(true_counts, generated_counts,plot_path='/', show_plot=False, R_error=False):
    pcc, p_value = pearsonr(true_counts, generated_counts)
    # mae = mean_absolute_error(true_counts, generated_counts)
    if R_error:
        relative_error = np.mean(np.abs(generated_counts - true_counts) / true_counts) * 100
        print(f"Relative Error: {relative_error:.2f}%")

    # 绘制散点图
    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.scatter(true_counts, generated_counts)
        plt.plot([0, 30], [0, 30], 'r--')
        plt.xlabel("gt Count")
        plt.ylabel("vs Count")
        plt.title(f"PCC={pcc:.2f}")

        # 绘制误差分布
        plt.subplot(122)
        plt.hist(np.array(generated_counts) - np.array(true_counts), bins=10)
        plt.axvline(0, color='r')
        plt.xlabel("Error (Generated - True)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path,'pcc-fig.jpg'))
        plt.show()
        plt.close()
    return pcc, p_value


def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2.
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    # print(len(pred_id_list))
    if len(pred_id_list) == 1:
        return 0

    true_masks = [None, ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None, ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    # print(aji_score)
    return aji_score


#############################################################################################################
def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).
    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.
    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    if len(pred_id_list) == 1:
        return [0, 0, 0], [0, 0, 0, 0]

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


#############################################################################################################
def get_dice_1(true, pred):
    """Traditional dice."""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    dice_score = 2.0 * np.sum(inter) / (np.sum(denom) + 0.0001)
    if np.sum(inter) == 0 and np.sum(denom) == 0:
        dice_score = 1  # to handel cases without any nuclei
    # print(dice_score)
    return dice_score


def calculate_mutual_information(img1, img2, bins=256):
    """计算两幅图像的互信息"""
    # 转换为灰度图并展平
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # 计算联合直方图
    joint_hist, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins)
    # 归一化
    joint_prob = joint_hist / np.sum(joint_hist)
    marginal_x = np.sum(joint_prob, axis=1)  # 图像1的边缘概率
    marginal_y = np.sum(joint_prob, axis=0)  # 图像2的边缘概率

    # 计算互信息（避免log(0)）
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if joint_prob[i, j] > 0 and marginal_x[i] > 0 and marginal_y[j] > 0:
                mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (marginal_x[i] * marginal_y[j]))
    return mi