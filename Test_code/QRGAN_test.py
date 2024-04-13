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
from utils.dataset import load_mnist, load_fmnist, denorm, select_from_dataset
from utils.datasets_2_padding import ImageDataset
from utils.wgan import compute_gradient_penalty
from models.QGCC_2_hybid_padding import PQWGAN_CC
from models.reverse import PQWGAN_CC3
from models.CQCC_wganRes import ClassicalGAN1
from models.CQCC_wganRes2 import ClassicalGAN2
from tqdm import trange
import time
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch import nn
from torchvision import transforms
from torch.autograd import Variable



def test(dataset_str, weight_dir, result_dir):
    device = torch.device('cpu')
    n_epochs = 1
    image_size = 32
    image_size_2 = 32
    channels = 1
    
    
    patches = 32
    layers = 12
    n_data_qubits = 5
    batch_size = 1
    checkpoint = 0
    patch_shape = (1, 32)

    ancillas = 0
    if n_data_qubits:
        qubits = n_data_qubits + ancillas
    else:
        qubits = math.ceil(math.log(image_size ** 2 // patches, 2)) + ancillas

    lr_G = 0.01
    lr_G_classical = 0.0002

    b1 = 0
    b2 = 0.9
    lambda_gp = 10
    n_critic = 2

    #####输出的文件路径##########
    out_dir = result_dir + "/result_A2B"
    out_dir_2 = result_dir + "/result_B2A"
    out_dir_3 = result_dir + "/result_A_real"
    out_dir_4 = result_dir + "/result_B_real"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_2, exist_ok=True)
    os.makedirs(out_dir_3, exist_ok=True)
    os.makedirs(out_dir_4, exist_ok=True)
    ############################################################

    #########调用QGQC_1###########################################

    gan_1 = PQWGAN_CC(image_size=image_size, image_size_2=image_size_2, channels=1, n_generators=patches,
                      n_qubits=qubits,
                      n_ancillas=ancillas, n_layers=layers, patch_shape=patch_shape)

    gan_2 = PQWGAN_CC3(image_size=image_size, image_size_2=image_size_2, channels=1, n_generators=patches,
                       n_qubits=qubits,
                       n_ancillas=ancillas, n_layers=layers, patch_shape=patch_shape)

    
    generator = gan_1.generator
    generator_2 = gan_2.generator

    ##################################加载数据#######################
    data = ImageDataset(dataset_str)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1)  # 数据集
    
    weight_files = os.listdir(weight_dir)
    weight_1 = ''
    weight_2 = ''
    for i in weight_files:
        if '_2' in i:
            weight_2 = i
            break
    generator.load_state_dict(torch.load(os.path.join(weight_dir, weight_2)))
    generator_2.load_state_dict(torch.load(os.path.join(weight_dir, weight_2)))

    j = 0
    for epoch in trange(n_epochs):
        for i, batch in enumerate(dataloader):
            
            real_A = batch['A']
            # Set model inputzzX
            real_B = batch['B']
            
            pred_images_B = generator(real_A)
            pred_images_A = generator_2(real_B)
            
            # 保存wassterin以及图片
            save_image(denorm(pred_images_B), os.path.join(out_dir, '{}.png'.format(j)), nrow=5)
            save_image(denorm(real_A), os.path.join(out_dir_3, '{}.png'.format(j)), nrow=5)
            
            save_image(denorm(pred_images_A), os.path.join(out_dir_2, '{}.png'.format(j)), nrow=5)
            save_image(denorm(real_B), os.path.join(out_dir_4, '{}.png'.format(j)), nrow=5)
            j = j + 1

            print("save image")
###############################主函数###########################################################
# if __name__ == "__main__":
#     file_index = [0, 1]
#     for i in file_index:
#         train(r"../Data/Canny/"+str(i), 32, 12, 5, 1, 0, (1, 32))
