import os
import numpy as np
import torch
from PIL import Image
import scipy.stats as stats
import matplotlib.pyplot as plt
from models import CellposeSegWrapper  # 导入你的Cellpose模型包装类
import torchvision
# --------------------------
# 1. 配置参数（根据你的环境修改）
# --------------------------
# 文件夹路径
gt_dir = './prostate_tissue/gt_stained'  # 真实值图像文件夹
vs_dir = './prostate_tissue/vs_stained'  # 生成图像文件夹
# 图像格式
img_ext = '.jpg'
# 设备配置（自动检测GPU/CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 图像预处理transform（需和你的模型训练时一致）
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 转为tensor并归一化到[0,1]
    # torchvision.transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],  # 常用ImageNet均值
    #     std=[0.229, 0.224, 0.225]    # 常用ImageNet标准差
    # )
])


# --------------------------
# 2. 核心工具函数
# --------------------------
def get_paired_image_paths(gt_dir, vs_dir, img_ext):
    """
    根据命名规则匹配gt和vs图像路径
    命名规则：12__24_gt_xxx.jpg <-> 12__24_vs_xxx.jpg
    返回：配对的(gt_path, vs_path)列表
    """
    # 获取所有gt/vs文件
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(img_ext)]
    vs_files = [f for f in os.listdir(vs_dir) if f.endswith(img_ext)]

    # 构建vs文件的映射字典：前缀（12__24）-> 完整文件名
    vs_file_map = {}
    for f in vs_files:
        # 提取vs文件的核心前缀（去掉_vs_及后面的部分）
        core_prefix = f.split('_vs_')[0]
        vs_file_map[core_prefix] = f

    # 匹配gt和vs文件
    paired_paths = []
    for gt_f in gt_files:
        # 提取gt文件的核心前缀（去掉_gt_及后面的部分）
        core_prefix = gt_f.split('_gt_')[0]
        # 查找对应的vs文件
        if core_prefix in vs_file_map:
            gt_path = os.path.join(gt_dir, gt_f)
            vs_path = os.path.join(vs_dir, vs_file_map[core_prefix])
            paired_paths.append((gt_path, vs_path))
            print(f"匹配成功：{gt_f} <-> {vs_file_map[core_prefix]}")
        else:
            print(f"未找到匹配的vs文件：{gt_f}")

    if not paired_paths:
        raise ValueError("未匹配到任何gt/vs图像对，请检查文件名规则！")
    return paired_paths


def process_image_with_cellpose(img_path, cel_model, transform, device):
    """
    加载图像并通过Cellpose模型提取概率密度图
    返回：概率密度图（numpy数组，shape=(H,W)）
    """
    try:
        # 加载图像并转换为RGB
        image = Image.open(img_path).convert("RGB")
        # 预处理并转为tensor
        img_tensor = transform(image).unsqueeze(0).to(device)  # [1,3,H,W]
        # 通过Cellpose模型推理
        with torch.no_grad():  # 禁用梯度，节省内存
            _, logits = cel_model(img_tensor)
        # 计算sigmoid得到概率密度
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H,W)
        return prob
    except Exception as e:
        print(f"处理图像失败 {img_path}：{str(e)}")
        raise


def calculate_pearson_for_pairs(paired_paths, cel_model, transform, device):
    """
    计算每对图像的Pearson相关系数
    返回：pearson_scores, 配对路径列表
    """
    pearson_scores = []
    valid_pairs = []

    # 初始化Cellpose模型
    cel_model = cel_model.to(device)
    cel_model.eval()  # 评估模式

    # 逐对处理
    for idx, (gt_path, vs_path) in enumerate(paired_paths):
        print(f"\n处理第{idx + 1}/{len(paired_paths)}对图像：")
        try:
            # 处理gt和vs图像，得到概率密度图
            gt_prob = process_image_with_cellpose(gt_path, cel_model, transform, device)
            vs_prob = process_image_with_cellpose(vs_path, cel_model, transform, device)

            # 展平为一维数组（计算整体分布的相关性）
            gt_flat = gt_prob.flatten()
            vs_flat = vs_prob.flatten()

            # 计算Pearson相关系数
            pearson_r, _ = stats.pearsonr(gt_flat, vs_flat)
            pearson_scores.append(pearson_r)
            valid_pairs.append((gt_path, vs_path))

            print(f"Pearson系数：{pearson_r:.4f}")
        except Exception as e:
            print(f"跳过该对图像：{str(e)}")
            continue

    if not pearson_scores:
        raise ValueError("没有有效图像对完成Pearson系数计算！")

    # 统计结果
    pearson_mean = np.mean(pearson_scores)
    pearson_std = np.std(pearson_scores)
    pearson_max = np.max(pearson_scores)
    pearson_min = np.min(pearson_scores)

    return pearson_scores, valid_pairs, pearson_mean, pearson_std, pearson_max, pearson_min


def visualize_pearson_results(pearson_scores, pearson_mean):
    """可视化Pearson系数分布"""
    plt.figure(figsize=(10, 6))
    plt.hist(pearson_scores, bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.axvline(pearson_mean, color='blue', linestyle='--', linewidth=2,
                label=f'均值: {pearson_mean:.4f}')
    plt.xlabel('Pearson相关系数', fontsize=12)
    plt.ylabel('图像对数量', fontsize=12)
    plt.title(f'Cellpose概率密度图Pearson系数分布 (共{len(pearson_scores)}对)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig('cellpose_pearson_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


# --------------------------
# 3. 主执行流程
# --------------------------
if __name__ == "__main__":

    # 步骤1：初始化Cellpose模型
    cel_model = CellposeSegWrapper()

    # 步骤2：匹配gt/vs图像对路径
    paired_paths = get_paired_image_paths(gt_dir, vs_dir, img_ext)

    # 步骤3：计算每对的Pearson系数
    pearson_scores, valid_pairs, mean_r, std_r, max_r, min_r = calculate_pearson_for_pairs(
        paired_paths, cel_model, transform, device
    )

    # 步骤4：输出统计结果
    print("\n===== Pearson系数统计结果 =====")
    print(f"有效图像对数量：{len(pearson_scores)}")
    print(f"Pearson系数均值：{mean_r:.4f} ± {std_r:.4f}")
    print(f"最大值：{max_r:.4f} | 最小值：{min_r:.4f}")

    # 步骤5：可视化结果
    visualize_pearson_results(pearson_scores, mean_r)

    # 可选：保存详细结果到txt
    with open('pearson_results.txt', 'w', encoding='utf-8') as f:
        f.write("Pearson系数详细结果\n")
        f.write("-" * 50 + "\n")
        for (gt_path, vs_path), r in zip(valid_pairs, pearson_scores):
            f.write(f"{os.path.basename(gt_path)} | {os.path.basename(vs_path)} | r={r:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"均值：{mean_r:.4f} ± {std_r:.4f}\n")
        f.write(f"最大值：{max_r:.4f} | 最小值：{min_r:.4f}\n")
    print("\n详细结果已保存到 pearson_results.txt")