import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


# 加载 .png 文件并转换为 float32 类型的 numpy 数组
def load_image_as_float32(filepath):
    image = Image.open(filepath).convert("RGB")

    return image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob("%s/unstained/*.*" % root))
        self.files_B = sorted(glob.glob("%s/stained/*.*" % root))

    def __getitem__(self, index):
        # seed = np.random.randint(2147483647)  # make a seed with numpy generator
        # print(seed)
        # 29  42  100
        seed = 100
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        item_a = self.transform(load_image_as_float32(self.files_A[index % len(self.files_A)]))
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        item_b = self.transform(load_image_as_float32(self.files_B[index % len(self.files_B)]))
        return {'A': item_a, 'B': item_b}

    def __len__(self):
        return len(self.files_A)


class ValDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob("%s/unstained/*" % root))
        self.files_B = sorted(glob.glob("%s/stained/*" % root))

    def __getitem__(self, index):

        item_a = self.transform(load_image_as_float32(self.files_A[index % len(self.files_A)]))
        item_b = self.transform(load_image_as_float32(self.files_B[index % len(self.files_B)]))

        return {'A': item_a, 'B': item_b, 'name_A':self.files_A[index % len(self.files_A)]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
