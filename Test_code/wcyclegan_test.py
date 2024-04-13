import os
import argparse
import math
import numpy as np
import torch
import itertools
from torch.utils.data import Subset, DataLoader
from PIL import Image
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.utils import save_image
# from torchvision.utils import save_image
from utils.dataset import load_mnist, load_fmnist, denorm, select_from_dataset
from utils.datasets_2_padding import ImageDataset
from utils.wgan import compute_gradient_penalty
# from models.CGCC import ClassicalGAN1
# from models.CGCC2 import ClassicalGAN2
from models.CQCC_wganRes import ClassicalGAN1
from models.CQCC_wganRes2 import ClassicalGAN2
from tqdm import trange

def test(dataset_dir, weight_dir, result_dir):
    
    batch_size = 1
    checkpoint = 0
    
    #####输出的文件路径##########
    # Create output dirs if they don't exist    
    out_dir = result_dir + "/result_A2B"
    out_dir_2 = result_dir + "/result_B2A"
    out_dir_3 = result_dir + "/result_A_real"
    out_dir_4 = result_dir + "/result_B_real"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_2, exist_ok=True)
    os.makedirs(out_dir_3, exist_ok=True)
    os.makedirs(out_dir_4, exist_ok=True)

    gan_1 = ClassicalGAN1()
    generator = gan_1.Generator()

    gan_2 = ClassicalGAN2()
    generator_2 = gan_2.Generator()

    data = ImageDataset(dataset_dir)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1)  # 数据集

    weight_files = os.listdir(weight_dir)
    weight_1 = ''
    weight_2 = ''
    for i in weight_files:
        if '_2' in i:
            weight_2 = i
            weight_1 = ''.join(weight_2.split('_2'))
            break
    generator.load_state_dict(torch.load(weight_dir  + '/' + weight_1))
    generator_2.load_state_dict(torch.load(weight_dir  + '/' + weight_2))

    j = 0
    for i, batch in enumerate(dataloader):

        real_A = batch['A']
        # Set model inputzzX
        real_B = batch['B']
        ############################生成器A2B的损失############################
        pred_images_B = generator(real_A)
        ############################生成器B2A的损失############################
        pred_images_A = generator_2(real_B)
        ############################生成器cycleABA的损失############################

        save_image(denorm(pred_images_B), os.path.join(out_dir, '{}.png'.format(j)), nrow=2)
        save_image(denorm(real_A), os.path.join(out_dir_3, '{}.png'.format(j)), nrow=2)

        save_image(denorm(pred_images_A), os.path.join(out_dir_2, '{}.png'.format(j)), nrow=2)
        save_image(denorm(real_B), os.path.join(out_dir_4, '{}.png'.format(j)), nrow=2)
        j = j + 1

    print("Saved  image")
###############################
# if __name__ == "__main__":
#     test("../Data/dilate", 1, 0)
