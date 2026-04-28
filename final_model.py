import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from tools.utils import get_config
import datetime
from tools.datasets import ImageDataset, ValDataset
from Model.generator import Gan_AttU_Net
from Model.discriminator import Discriminator_att
from tools.utils import Resize
import losses
from Model.reg import Reg1
from tools.transformer import Transformer_2D
from Segmentation.models import CellposeSegWrapper
from Segmentation.SegLoss import segmentation_loss

from draw_plot import draw_loss
from evaluation import MAE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class virtual_stain_model():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.exp_name_title = 'G:G-att-Unet,D:D-att;R:R1;Seg:dice+F1(dim_12)'
        # attunet -Gan and special D
        self.netG_A2B = Gan_AttU_Net(config['input_nc'], config['output_nc'],type='instanceNorm').cuda()
        self.netD_B = Discriminator_att(config['input_nc'],scale=2).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        #
        if config['reg']:
            self.R_A = Reg1(config['size'], config['size'], config['input_nc'], config['input_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        # 分割模型
        if config['seg']:
            self.cellpose_net = CellposeSegWrapper()

        # Lossess
        self.MSE_loss = losses.mse_loss()
        self.BCE_loss = losses.bce_loss()
        self.L1_loss = losses.n1_loss()
        self.ncc_loss = losses.NCC(win=20, eps=1e-3)
        self.grad_loss = losses.smooothing_loss
        self.Seg_loss = segmentation_loss
        # Inputs & targets memory allocation

        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.input_A_test = Tensor(config['batchSize_val'], config['input_nc'], config['val_size'], config['val_size'])
        self.input_B_test = Tensor(config['batchSize_val'], config['output_nc'], config['val_size'], config['val_size'])

        # Dataset loader
        train_transforms = [transforms.ToTensor(), Resize(size_tuple=(config['size'], config['size']))]
        val_transforms = [transforms.ToTensor(),Resize(size_tuple=(config['val_size'], config['val_size']))]

        self.dataloader = DataLoader(ImageDataset(config['dataRoot'], transforms_=train_transforms),
                                     batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'], drop_last=True)

        self.val_data = DataLoader(ValDataset(config['val_dataRoot'], transforms_=val_transforms),
                                   batch_size=1, shuffle=False, num_workers=config['n_cpu'], drop_last=True)

    def train(self, cfg_content):
        total_loss_mean = []
        reg_loss_mean, dvf_loss_mean = [], []  # 形变矢量场的loss
        tran_loss_mean, adv_loss_mean = [], [] # generator loss
        loss_dis_mean = []   # discriminator loss
        loss_con_seg = []  # consistency-seg loss
        total_datasets = len(self.dataloader) * self.config['batchSize']
        print(f'total train dataset,     total val dataset = {total_datasets}  {len(self.val_data)}')
        epoch_loops = self.config['n_epochs']
        exp_res_dir = os.path.join(self.config["save_root"], cfg_content.split(':')[1][2:])
        if not os.path.exists(exp_res_dir):
            os.makedirs(exp_res_dir)

        ###### Training ######
        print('time',datetime.datetime.now(),'exp: ', self.exp_name_title)
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            print(f'----------------epoch={epoch+1}/{epoch_loops}----------------')
            loss1, loss2, loss3, loss4, lossD, loss_Seg = 0, 0, 0, 0, 0, 0
            for i, batch in enumerate(self.dataloader):
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                self.optimizer_R_A.zero_grad()
                self.optimizer_G.zero_grad()
                #### regist sys loss
                fake_B = self.netG_A2B(real_A)
                Trans = self.R_A(fake_B, real_B)
                SysRegist_A2B = self.spatial_transform(fake_B, Trans)

                SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B)  ###SR
                pred_fake0 = self.netD_B(fake_B)
                adv_loss = self.config['Adv_lamda'] * self.BCE_loss(pred_fake0, torch.ones_like(pred_fake0).cuda())

                ####smooth loss  这里是dvf的损失
                SM_loss = self.config['Smooth_lamda'] * self.grad_loss(Trans)

                if epoch > (self.config['n_epochs'] / 5):
                    vs_seg = self.cellpose_net(SysRegist_A2B)
                    gt_seg = self.cellpose_net(real_B.detach())
                    #  余弦相似度损失 和 其他损失
                    loss_seg1, loss_seg2 = self.Seg_loss(vs_seg, gt_seg)
                    loss_seg = (loss_seg1 + loss_seg2) * self.config['Seg_lamda']
                else:
                    loss_seg = torch.tensor(0)

                # 总损失
                total_loss = adv_loss + SR_loss + SM_loss + loss_seg

                total_loss.backward()
                self.optimizer_R_A.step()
                self.optimizer_G.step()
                self.optimizer_D_B.zero_grad()

                # 更新 判别器 D
                with torch.no_grad():
                    fake_B = self.netG_A2B(real_A)
                pred_fake0 = self.netD_B(fake_B)
                pred_real = self.netD_B(real_B)
                loss_D_B = self.config['Adv_lamda'] * (self.BCE_loss(pred_fake0, torch.zeros_like(pred_fake0).cuda()) +
                                                       self.BCE_loss(pred_real, torch.ones_like(pred_real).cuda()))
                # 更新 判别器 D
                loss_D_B.backward()
                self.optimizer_D_B.step()

                loss1 += total_loss.item()   # total loss
                loss2 += (SM_loss.item() + SR_loss.item())  # Reg loss 整体的配准损失
                loss3 += SM_loss.item()  # dvf loss///////
                loss2 += SR_loss.item()  #
                loss4 += adv_loss.item()  # adv loss
                loss_Seg += loss_seg.item()

                if i % 40 == 0:
                    print(f'total loss:{total_loss} SM_loss:{SM_loss} adv_loss:{adv_loss} D_loss: {loss_D_B} SR_loss:{SR_loss},Seg:{loss_seg}')

            total_loss_mean.append(loss1 / total_datasets)
            reg_loss_mean.append(loss2 / total_datasets)
            dvf_loss_mean.append(loss3 / total_datasets)
            tran_loss_mean.append(loss2 / total_datasets)  # 图像的转化损失
            adv_loss_mean.append(loss4  / total_datasets)
            loss_dis_mean.append(lossD / total_datasets)
            loss_con_seg.append(loss_Seg / total_datasets)
            # Save models checkpoints
            if epoch >= (self.config['n_epochs'] / 5):
                torch.save(self.netG_A2B.state_dict(), exp_res_dir + f'/epoch={epoch}_netG_A2B.pth')

            #############val###############
            with torch.no_grad():
                MAE_total = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A_test.copy_(batch['A']))
                    real_B = Variable(self.input_B_test.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    mae = MAE(fake_B, real_B)
                    MAE_total += mae
                    num += 1

                print('Val MAE:', MAE_total / num)

        draw_loss(self.exp_name_title,total_loss_mean, 'total_loss',exp_res_dir)
        draw_loss(self.exp_name_title, reg_loss_mean, 'Registration loss', exp_res_dir)
        draw_loss(self.exp_name_title,dvf_loss_mean, 'dvf_smooth_loss',exp_res_dir)
        draw_loss(self.exp_name_title,tran_loss_mean, 'transformation loss',exp_res_dir)
        draw_loss(self.exp_name_title,adv_loss_mean, 'G-loss',exp_res_dir)
        draw_loss(self.exp_name_title,loss_dis_mean, 'D-loss',exp_res_dir)
        draw_loss(self.exp_name_title, loss_con_seg, 'Adversary Segmentation', exp_res_dir)

        with open(f"{exp_res_dir}/cfg_readme.txt", "a", encoding="utf-8") as file:
            file.write(cfg_content)


def log_metrics(file_path, metrics):
    """将指标追加到日志文件"""
    with open(file_path, 'a', encoding='utf-8') as f:  # 'a' 表示追加模式
        f.write(f"{metrics}\n")  # 直接写入字典或字符串


if '__main__' == __name__:
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    cfg = (f'NC+R6:说明final:探究实验，使用相同数据sample22 512size 2w张 ,判别器D似乎太强了，需要适当调整 time:{formatted_time}'
           '\n 验证sample24 2048图像. 实验参数同exp19 epoch:40,batch_size:8,'
           '使用R网络，使用分割损失 seed=100 adv loss :torch.mean();\n'
           '分割损失使用 soft + dice；D 模型filter减半\n'
           '--------------------------------------------------------------\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/Gan_baseline_Seg.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    Trainer = virtual_stain_model(config=config)
    sdir = Trainer.config['save_root']
    logtxt = os.path.join(sdir,'config.txt')

    exp = 'train'
    Trainer.train(cfg_content=cfg)

