import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        # self.transform = transforms.Compose(transforms_)
        self.transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.unaligned = unaligned

        # self.files_A = sorted(glob.glob(os.path.join(root, '%s/out1216' % mode) + '/*.*'))
        # self.files_B = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_A = sorted(glob.glob(os.path.join(root, 'B', '*.png')))
        self.files_B = sorted(glob.glob(os.path.join(root, 'A', '*.png')))

    def __getitem__(self, index):
        item_A = Image.open(self.files_A[index])
        item_A = self.transform(item_A)

        item_B = Image.open(self.files_B[index])
        item_B = self.transform(item_B)

        return {'B': item_A, 'A': item_B}

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))