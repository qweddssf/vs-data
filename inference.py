import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from tools.utils import get_config
import datetime
from tools.datasets import ValDataset
from Segmentation.evaluation import calculate_nuclei, cal_pcc,extractSeg,get_f1_score
from Model.generator import Gan_AttU_Net, UNetGenerator
from tools.utils import Resize
import numpy as np
import matplotlib.pyplot as plt
from draw_plot import draw_violin_statistic_cur
from tqdm import tqdm
from evaluation import SSIM,MAE,PSNR
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

isPrintRes = True
isStaticRes = True
isSaveRes = False
class InferenceNetwork():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_eval, self.nuclei_evl = self.config['base_eval'], self.config['nuclei_eval']
        self.netG_A2B = Gan_AttU_Net(config['input_nc'], config['output_nc'],type='instanceNorm',scale_factor=1).cuda()

        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A_test = Tensor(config['batchSize_val'], config['input_nc'], config['val_size'], config['val_size'])
        self.input_B_test = Tensor(config['batchSize_val'], config['output_nc'], config['val_size'], config['val_size'])

        val_transforms = [transforms.ToTensor(), Resize(size_tuple=(config['val_size'], config['val_size']))]
        self.val_data = DataLoader(ValDataset(config['val_dataset_root'].replace('tissue_sec',config['tissue_sec']), transforms_=val_transforms),
                                   batch_size=1, shuffle=False, num_workers=config['n_cpu'], drop_last=True)

    def test(self, weight_root, epoch, output_dir,logtxt):
        print('load weight name', weight_root)
        print('save dir',output_dir)
        img_dir = os.path.join(output_dir, 'images')

        if self.config['cuda']:
            self.netG_A2B.load_state_dict(torch.load(weight_root))
        else:
            state_dict = torch.load(weight_root, map_location=torch.device('cpu'))
            self.netG_A2B.load_state_dict(state_dict)
            # 确保模型本身也运行在CPU上
            self.netG_A2B = self.netG_A2B.to(torch.device('cpu'))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(img_dir)
        with torch.no_grad():
            mae_list = []
            psnr_list = []
            ssim_list = []
            nuclei_cnt_gt = []
            nuclei_cnt_vs = []
            f1_list, pre_list, recall_list = [], [], []
            print(f'total dataset {len(self.val_data)}')
            start_time = time.time()
            for batch in tqdm(self.val_data):
                real_A = Variable(self.input_A_test.copy_(batch['A']))
                real_B = Variable(self.input_B_test.copy_(batch['B'])).detach().cpu().numpy().squeeze()

                fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                fake_B = np.clip(fake_B, 0, 1)
                real_B = np.clip(real_B, 0, 1)
                name_A = batch['name_A'][0].split('/')[-1]
                output_path = os.path.join(img_dir, name_A.replace('AF', 'combined'))

                if isPrintRes:
                    real_A = np.clip(real_A.detach().cpu().numpy().squeeze(), 0, 1)
                    self.save_image(np.transpose(real_A, [1, 2, 0]),
                                    np.transpose(real_B, [1, 2, 0]),
                                    np.transpose(fake_B, [1, 2, 0]),output_path)

                if isSaveRes:
                    # 是否保存虚拟染色结果
                    plt.imsave(output_path.replace('combined','vs'), np.transpose(fake_B, [2, 1, 0]))
                    # plt.imsave(output_path.replace('combined','gt'), np.transpose(real_B, [2, 1, 0]))

                if self.nuclei_evl:
                    # 是否结算核指标 pcc
                    real_mask, _ = extractSeg(np.transpose(real_B, [1, 2, 0]))
                    nuclei_cnt_gt.append(calculate_nuclei(real_mask))
                    fake_mask, _ = extractSeg(np.transpose(fake_B, [1, 2, 0]))
                    nuclei_cnt_vs.append(calculate_nuclei(fake_mask))
                    # f1
                    # f1, precision, recall = get_f1_score(fake_mask, real_mask)
                    # f1_list.append(f1)
                    # pre_list.append(precision)
                    # recall_list.append(recall)

                if self.base_eval:
                    # 是否计算基础的生成图像评估指标
                    mae = MAE(fake_B, real_B)
                    psnr = PSNR(fake_B, real_B)
                    mssim = SSIM(real_B, fake_B)
                    mae_list.append(mae)
                    psnr_list.append(psnr)
                    ssim_list.append(mssim)

            total_time = time.time() - start_time
            print(f"\n总运行时间: {total_time:.2f} 秒")
            if isStaticRes:
                # 是否保存打印的统计图结果
                draw_violin_statistic_cur(mae_list, f'{epoch}_MAE', output_dir)
                draw_violin_statistic_cur(psnr_list, f'{epoch}_PSNR', output_dir)
                draw_violin_statistic_cur(ssim_list, f'{epoch}_SSIM', output_dir)

            now = datetime.datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            res_dic = {'Time': formatted_time, 'epoch': epoch}
            if self.nuclei_evl:
                pcc_value = cal_pcc(nuclei_cnt_gt, nuclei_cnt_vs, plot_path=output_dir, show_plot=True)
                res_dic.update({'f1 mean': np.mean(f1_list),'precision mean': np.mean(pre_list),'recall mean': np.mean(recall_list),
                                'pcc': pcc_value[0]})
                print('f1 mean', np.mean(f1_list))
                print('precision mean', np.mean(pre_list))
                print('recall mean', np.mean(recall_list))
                print('PCC', pcc_value[0])

            if self.base_eval:
                res_dic.update({'MAE': np.mean(mae_list), 'PSNR:': np.mean(psnr_list), 'SSIM': np.mean(ssim_list)})
                print('MAE:', np.mean(mae_list))
                print('PSNR:', np.mean(psnr_list))
                print('SSIM:', np.mean(ssim_list))
            print(res_dic)
            log_metrics(logtxt, res_dic)


    def save_image(self, real_A, real_B, fake_B, dir):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1行3列的子图

        # 转换图像数据并显示
        axes[0].imshow(fake_B)
        axes[0].set_title('virtually stained')
        axes[0].axis('off')  # 关闭坐标轴

        axes[1].imshow(real_B)
        axes[1].set_title('ground truth')
        axes[1].axis('off')

        axes[2].imshow(real_A)
        axes[2].set_title('unstained')
        axes[2].axis('off')

        plt.savefig(dir)
        plt.tight_layout()
        plt.close(fig)

def log_metrics(file_path, metrics):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"{metrics}\n")


if '__main__' == __name__:
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    cfg = (f'cfg：'
           '--------------------------------------------------------------\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/inference.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)

    model = InferenceNetwork(config=config)
    sdir = config['save_root'].replace('exp_name',config['exp_name']).replace('tissue_sec',config['tissue_sec'])
    # evl_dir = config['eva_root'].replace('exp_name',config['exp_name']).replace('tissue_sec',config['tissue_sec'])
    weight_dir = config['weight_path'].replace('tissue_sec',config['tissue_sec']).replace('exp_name',config['exp_name'])
    logtxt = os.path.join(sdir, 'config.txt')


    def test_certain_epoch(epoch):
        print(f'--load weight epoch--{epoch}-----------------')
        print('val dataset：', model.config['val_dataset_root'])
        sample_code = model.config['val_dataset_root'].split('/')[-2]
        weight_pth_id = f'epoch={epoch}_netG_A2B.pth'

        weight_root = os.path.join(weight_dir, weight_pth_id)
        output_dir = os.path.join(sdir, f'epoch={epoch}')

        # logtxt = os.path.join(evl_dir, f'test_res.txt')
        model.test(weight_root=weight_root, epoch=epoch, logtxt=logtxt, output_dir=output_dir)
        # model.test(weight_root=weight_root, epoch=epoch, output_dir=output_dir)

    best_epoch = model.config['best_epochs']

    for be in best_epoch:
        test_certain_epoch(be)


