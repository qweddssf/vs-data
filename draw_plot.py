import os
import matplotlib.pyplot as plt

def draw_violin_statistic_cur(data, type, dir, cur_type='evaluation_cur'):
    # 画箱线图和小提琴图
    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # 绘制箱线图
    sns.boxplot(data=data, ax=axes[0])
    axes[0].set_title(f'{type} box plots')
    # axes[0].set_xticklabels(type)

    # 绘制小提琴图
    sns.violinplot(data=data, ax=axes[1])
    axes[1].set_title(f'{type} violin plots')
    save_path = os.path.join(dir, cur_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'{type}_box_violin_plot.jpg'), format='jpg', dpi=300)
    plt.close()

def draw_loss(exp_name, loss_list, lose_type,save_dir):
    if(len(loss_list) == 0): return
    epoch = range(0, len(loss_list))
    plt.figure()
    plt.plot(epoch, loss_list)
    plt.title(exp_name)
    plt.xlabel('epoch')
    plt.ylabel(lose_type)
    plt.savefig(os.path.join(save_dir, f'{lose_type}-epoch.png'))
    plt.close()

