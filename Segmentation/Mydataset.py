import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch



class NucleiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = image_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 读取图像和掩码
        img_path = self.image_names[idx]
        mask_path = self.image_names[idx]

        image = np.array(Image.open(img_path).convert("RGB"))  # 转为numpy数组
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # 转为灰度

        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 归一化并转为Tensor
        mask = mask / 255.0  # 将掩码值缩放到[0, 1]
        return image, mask.unsqueeze(0)  # 增加通道维度

# 数据增强配置
