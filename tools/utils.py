import random
import time
import datetime
import sys
import yaml
from torch.autograd import Variable
import torch
from visdom import Visdom
import torch.nn.functional as F
import numpy as np
import os
class Resize():
    def __init__(self, size_tuple, use_cv = True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv


    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)
 
        tensor = F.interpolate(tensor, size = [self.size_tuple[0],self.size_tuple[1]])

        tensor = tensor.squeeze(0)
 
        return tensor#1, 64, 128, 128
class ToTensor():
    def __call__(self, tensor):
        tensor = np.expand_dims(tensor, 0)
        return torch.from_numpy(tensor)

def tensor2image(tensor):
    image = (127.5*(tensor.cpu().float().numpy()))+127.5
    image1 = image[0]
    for i in range(1,tensor.shape[0]):
        image1 = np.hstack((image1,image[i]))
    
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    #print ('image1.shape:',image1.shape)
    return image1.astype(np.uint8)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    # print ('m:',m)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
        
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


# 用来解压.tar文件
def getTarfileImgs(tgt_path,save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    origin_cnt = getFileLength(save_path)

    import tarfile
    with tarfile.open(tgt_path,'r') as tar:
        tar.extractall(path=save_path)
    print(f'{tgt_path} 文件已经解压到---> {save_path}')
    cur_cnt = getFileLength(save_path)
    print(f'原有文件: {origin_cnt}')
    print(f'现有文件: {cur_cnt}')
    print(f'解压得到文件：{cur_cnt - origin_cnt}')

def getFileLength(path):
    import glob
    files = glob.glob(path)
    if path[-4:] != '.jpg':
        files = glob.glob(os.path.join(path,'*.jpg'))
    return len(files)


def compute_gradient_penalty(netD, real_samples, fake_samples, device):
    """计算梯度惩罚项（WGAN-GP）"""
    # 生成随机插值样本
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # 计算判别器对插值样本的输出（未经过Sigmoid）
    d_interpolates = netD(interpolates)

    # 创建虚拟梯度（用于自动微分）
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # 计算梯度范数并惩罚偏离1的部分
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def remove_folder(folder_path):
    import shutil
    try:
        # 可以删除包含文件和子文件夹的文件夹
        shutil.rmtree(folder_path)
        print(f"文件夹 {folder_path} 及其内容已成功删除")
    except OSError as e:
        print(f"删除文件夹时出错: {e}")


if __name__ == '__main__':

    for num in [24]:
        type = 'test'
        size = 2048
        stride = 2048
        overlap = 0
        # 训练集的解压
        if type == 'train':
            st_tgt_path = f'../dataset/tar_512_files/sample_{num}_stained.tar'
            us_tgt_path = f'../dataset/tar_512_files/sample_{num}_unstained.tar'
            st_save_path = f'../dataset/train2/sample_{num}/stained/'
            us_save_path = f'../dataset/train2/sample_{num}/unstained/'
        else:
            # 测试集的解压
            st_tgt_path = f'../dataset/tar_{size}_files/stride{stride}_overlap{overlap}/sample_{num}_stained.tar'
            us_tgt_path = f'../dataset/tar_{size}_files/stride{stride}_overlap{overlap}/sample_{num}_unstained.tar'
            st_save_path = f'../dataset/size{size}/stride{stride}_overlap{overlap}/sample_{num}/stained/'
            us_save_path = f'../dataset/size{size}/stride{stride}_overlap{overlap}/sample_{num}/unstained/'

        # remove_folder(st_save_path)
        # remove_folder(us_save_path)
        # print(getFileLength(st_save_path),getFileLength(us_save_path))

        print(f'处理文件：sample_{num}')
        print('染色后图像')
        getTarfileImgs(st_tgt_path, st_save_path)

        print('染色前图像')
        getTarfileImgs(us_tgt_path,us_save_path)
        print(f'sample_{num}', getFileLength(st_save_path))


    # 删除文件夹
    # # remove_folder(st_save_path)
    # # remove_folder(us_save_path)
    # print(getFileLength(st_save_path),getFileLength(us_save_path))



    # print(f'处理文件：sample_{num}')
    # print('染色后图像')
    #
    # getTarfileImgs(st_tgt_path, st_save_path)
    # print('染色前图像')
    # getTarfileImgs(us_tgt_path,us_save_path)

