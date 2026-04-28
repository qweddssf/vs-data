import glob
import os.path

import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
from cellpose import models, io
from cellpose.utils import masks_to_outlines  # Cellpose工具函数
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
model = models.Cellpose(gpu=True, model_type='cyto')

#############################################################################################################
def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


#############################################################################################################
# def pair_coordinates(setA, setB, radius):
#     """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
#     unique pairing (largest possible match) when pairing points in set B
#     against points in set A, using distance as cost function.
#     Args:
#         setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
#                     of N different points
#         radius: valid area around a point in setA to consider
#                 a given coordinate in setB a candidate for match
#     Return:
#         pairing: pairing is an array of indices
#         where point at index pairing[0] in set A paired with point
#         in set B at index pairing[1]
#         unparedA, unpairedB: remaining poitn in set A and set B unpaired
#     """
#     # * Euclidean distance as the cost matrix
#     pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')
#
#     # * Munkres pairing with scipy library
#     # the algorithm return (row indices, matched column indices)
#     # if there is multiple same cost in a row, index of first occurence
#     # is return, thus the unique pairing is ensured
#     indicesA, paired_indicesB = linear_sum_as/signment(pair_distance)
#
#     # extract the paired cost and remove instances
#     # outside of designated radius
#     pair_cost = pair_distance[indicesA, paired_indicesB]
#
#     pairedA = indicesA[pair_cost <= radius]
#     pairedB = paired_indicesB[pair_cost <= radius]
#
#     pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
#     unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
#     unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
#     return pairing, unpairedA, unpairedB

def extractSeg(image,diameter=38, channels=[0,0]):
    masks, flows, styles, _ = model.eval(
        image,
        diameter=diameter,  # 预估核直径（可选）
        channels=channels
    )
    return masks, flows
# def calculate_nuclei(image, diameter=38, channels=[0,0]):
#     # model = models.Cellpose(gpu=True, model_type='cyto')
#     masks, _ = extractSeg(image, diameter, channels)
#     n_nuclei = masks.max()
#     return n_nuclei

def calculate_nuclei(masks):
    # model = models.Cellpose(gpu=True, model_type='cyto')
    # masks, _ = extractSeg(image, diameter, channels)
    n_nuclei = masks.max()
    return n_nuclei


###########  计算Tp FP FN F1

# 提取每个细胞核的质心坐标


def calculate_tp_fp_fn(virtual_mask, real_mask, iou_threshold=0.0001):
    """基于掩码IoU的TP、FP、FN计算"""
    # 获取虚拟核和真实核的标签（排除背景0）
    virtual_labels = np.unique(virtual_mask)[1:]  # 虚拟核标签：1,2,...
    real_labels = np.unique(real_mask)[1:]  # 真实核标签：1,2,...
    tp = 0
    matched_real = set()
    # 遍历每个虚拟核
    for v_label in virtual_labels:
        v_mask = (virtual_mask == v_label).astype(np.uint8)  # 虚拟核的掩码
        max_iou = 0
        best_r_label = -1
        # 与每个真实核计算IoU
        for r_label in real_labels:
            if r_label in matched_real:
                continue  # 跳过已匹配的真实核
            r_mask = (real_mask == r_label).astype(np.uint8)  # 真实核的掩码
            intersection = np.sum(v_mask & r_mask)
            union = np.sum(v_mask | r_mask)
            if union == 0:
                iou = 0
            else:
                iou = intersection / union

            if iou > max_iou:
                max_iou = iou
                best_r_label = r_label

        # 若最大IoU超过阈值，则判定为TP
        if max_iou >= iou_threshold:
            tp += 1
            matched_real.add(best_r_label)

    fp = len(virtual_labels) - tp  # 虚拟核总数 - TP
    fn = len(real_labels) - len(matched_real)  # 真实核总数 - 匹配的真实核
    return tp, fp, fn

# 计算F1值
def calculate_f1(tp, fp, fn):
    if tp == 0:  # 避免除以0
        return 0.0,0.0,0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1,precision,recall


def get_f1_score(vs_mask, gt_mask, iou_threshold=0.0001):
    tp,fp,fn = calculate_tp_fp_fn(vs_mask, gt_mask, iou_threshold)
    f1, precision,recall = calculate_f1(tp,fp,fn)
    return f1,precision,recall

def get_f1_score2(virtual_mask, real_mask):
    tp, fp, fn = calculate_pixel_level_metrics(virtual_mask, real_mask)
    f1, precision,recall = calculate_f1(tp,fp,fn)
    return f1,precision,recall

# 基于 “像素级” 的 F1 计算（整体分割评价）
def calculate_pixel_level_metrics(virtual_mask, real_mask):
    """像素级的TP、FP、FN计算"""
    # 将掩码转为二值图（0=背景，1=核）
    virtual_binary = (virtual_mask > 0).astype(np.uint8)
    real_binary = (real_mask > 0).astype(np.uint8)

    tp = np.sum(virtual_binary & real_binary)  # 均为核的像素
    fp = np.sum(virtual_binary & (1 - real_binary))  # 虚拟核、真实背景
    fn = np.sum((1 - virtual_binary) & real_binary)  # 虚拟背景、真实核
    return tp, fp, fn
###############
def cal_num_nuclei_by_path(image_path):
    # if image is None:
    image = io.imread(image_path)  # 支持 TIFF/PNG/JPG
    # 初始化模型（使用细胞核专用模型）
    # model = models.Cellpose(model_type="nuclei")
    n_nuclei = calculate_nuclei(image)
    return n_nuclei

# calculate pcc mae 并且绘制散点图

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

def pcc_evaluation(gt_img_path,pcc_plot_path,replace_str='unstained'):
    gt_img_list = glob.glob(gt_img_path)
    true_counts = []
    pred_counts = []
    for path in gt_img_list:
        gt_nuclei_cnt = cal_num_nuclei_by_path(path)
        vs_nuclei_cnt = cal_num_nuclei_by_path(path.replace('stained', replace_str))
        true_counts.append(gt_nuclei_cnt)
        pred_counts.append(vs_nuclei_cnt)
    return cal_pcc(true_counts, pred_counts, plot_path=pcc_plot_path)


def calculate_nucleus_sizes(mask):
    """
    从Cellpose输出的掩码中计算每个细胞核的大小（像素数量）
    参数:
        mask: Cellpose输出的分割掩码（二维数组，每个细胞核用唯一整数标记）
    返回:
        sizes: 所有细胞核的大小列表（像素数）
        avg_size: 平均细胞核大小（像素数）
    """
    # 获取所有细胞核的标签（排除背景0）
    labels = np.unique(mask)
    labels = labels[labels != 0]  # 过滤背景

    if len(labels) == 0:
        return [], 0.0  # 无细胞核

    # 计算每个细胞核的像素数量（大小）
    sizes = []
    for label in labels:
        # 统计当前标签对应的像素数量
        nucleus_pixels = np.sum(mask == label)
        sizes.append(nucleus_pixels)
    avg_size = np.mean(sizes)
    return sizes, avg_size


def compare_nucleus_sizes(virtual_mask, real_mask, pixel_size=None):
    """
    对比虚拟染色与真实染色的细胞核大小

    参数:
        virtual_mask: 虚拟染色图像的Cellpose分割掩码
        real_mask: 真实染色图像的Cellpose分割掩码
        pixel_size: 单个像素的物理尺寸（如μm/像素，可选）

    返回:
        包含大小统计信息的字典
    """
    # 计算大小
    virtual_sizes, virtual_avg = calculate_nucleus_sizes(virtual_mask)
    real_sizes, real_avg = calculate_nucleus_sizes(real_mask)

    # 转换为物理尺寸（若提供pixel_size）
    unit = "pixel"
    if pixel_size is not None:
        virtual_sizes = [s * (pixel_size ** 2) for s in virtual_sizes]  # 面积=像素数×(像素尺寸)^2
        real_sizes = [s * (pixel_size ** 2) for s in real_sizes]
        virtual_avg *= (pixel_size ** 2)
        real_avg *= (pixel_size ** 2)
        unit = f"μm²"  # 假设pixel_size单位为μm

    # 计算差异指标
    size_diff = virtual_avg - real_avg  # 平均大小差异
    size_ratio = virtual_avg / real_avg if real_avg != 0 else 0.0  # 平均大小比率（虚拟/真实）

    return {
        "virtual": {
            "sizes": virtual_sizes,
            "average": virtual_avg,
            "count": len(virtual_sizes)
        },
        "real": {
            "sizes": real_sizes,
            "average": real_avg,
            "count": len(real_sizes)
        },
        "difference": size_diff,  # 虚拟-真实
        "ratio": size_ratio,  # 虚拟/真实
        "unit": unit
    }


if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.ToTensor()
    vs_path = 'test_img/prostate_tissue/vs_stained/sample_12_unstained_3_3.jpg'
    gt_path = 'test_img/prostate_tissue/gt_stained/sample_12_stained_3_3.jpg'
    image_vs1 = transform(Image.open(vs_path).convert("RGB"))
    image_gt1 = transform(Image.open(gt_path).convert("RGB"))
    mask_gt,_ = extractSeg(image_gt1)
    mask_vs,_ = extractSeg(image_vs1)
    print('gt',calculate_nucleus_sizes(mask_gt))
    print('vs',calculate_nucleus_sizes(mask_vs))
    # # 计算两细胞核数量的
    # print('vs ', cal_num_nuclei_by_path(vs_path))
    # print('gt ', cal_num_nuclei_by_path(gt_path))


    # exp_name = '28'
    # path = f'../output/G_att_R2_seg/NC+R6/{exp_name}/img/*.jpg'
    # gt_img_list = glob.glob(path)
    # filtered_list = []
    # for img in gt_img_list:
    #     if 'combine' not in img and 'gt' in img:
    #         filtered_list.append(img)
    #
    # true_counts = []
    # pred_counts = []
    # for path in tqdm(filtered_list):
    #     gt_nuclei_cnt = cal_num_nuclei_by_path(path)
    #     vs_nuclei_cnt = cal_num_nuclei_by_path(path.replace('gt', 'vs'))
    #     true_counts.append(gt_nuclei_cnt)
    #     pred_counts.append(vs_nuclei_cnt)
    #
    # cal_pcc(true_counts, pred_counts, plot_path=f'../output/G_att_R2_seg/NC+R6/{exp_name}/',show_plot=True)
    pass