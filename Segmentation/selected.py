import os
import shutil


def find_and_copy_vs_files(gt_folder, vs_folder, operation="copy"):
    """
    从gt文件夹遍历所有图片，自动匹配vs图片并复制/移动到gt文件夹
    :param gt_folder: gt图片所在文件夹（gt_stained）
    :param vs_folder: 待查找的vs图片文件夹（虚拟染色文件夹）
    :param operation: copy=复制  move=移动
    """
    # 统计变量
    total_gt = 0
    found_vs = 0
    not_found = 0

    # 遍历 gt 文件夹所有文件
    for filename in os.listdir(gt_folder):
        file_path = os.path.join(gt_folder, filename)

        # 只处理文件，跳过文件夹
        if not os.path.isfile(file_path):
            continue

        # 只处理包含 _gt_ 的图片（你的命名规则）
        if "_gt_" not in filename:
            continue

        total_gt += 1
        print(f"正在匹配: {filename}")

        # ✅ 核心：把 gt 替换成 vs，生成配对的vs文件名
        vs_filename = filename.replace("_gt_", "_vs_")
        vs_file_path = os.path.join(vs_folder, vs_filename)

        # 检查vs文件是否存在
        if os.path.exists(vs_file_path):
            # 目标路径：直接放在 gt_stained 文件夹里
            target_path = os.path.join(gt_folder, vs_filename)

            # 复制/移动
            if operation == "copy":
                shutil.copy2(vs_file_path, target_path)
            else:
                shutil.move(vs_file_path, target_path)

            found_vs += 1
            print(f"  → 找到配对: {vs_filename} (已{operation}到gt文件夹)")
        else:
            not_found += 1
            print(f"  → 未找到: {vs_filename}")

    # 输出结果
    print("\n" + "=" * 50)
    print(f"扫描完成！")
    print(f"GT图片总数: {total_gt}")
    print(f"成功匹配VS图片: {found_vs}")
    print(f"未找到配对: {not_found}")
    print(f"所有VS图片已保存到: {gt_folder}")
    print("=" * 50)


# ====================== 【只需要修改这里】 ======================
if __name__ == "__main__":
    # 你的 gt 图片文件夹（固定不变）
    GT_FOLDER = r"./prostate_tissue/gt_stained"

    # 你的 虚拟染色vs文件夹（改成你自己的路径）
    VS_FOLDER = "../output/G_att_R2_seg/NC+R6/28/img"

    # 操作：copy=复制  move=移动
    OPERATION = "copy"
    # ====================================================================

    # 运行程序
    find_and_copy_vs_files(GT_FOLDER, VS_FOLDER, OPERATION)