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


from models.CQCC_wganRes import ClassicalGAN1
from models.CQCC_wganRes2 import ClassicalGAN2

from tqdm import trange


def test(dataset_str, weight_dir, result_dir):

    batch_size = 1
    checkpoint = 0
    n_epochs = 1
   
    
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

########################################################
    channels=1
  
    gan_1 = ClassicalGAN1()
    generator = gan_1.Generator()
 


    # 计算生成器的参数数量
    total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    # total_params = sum(p.numel() for p in gan_1.ClassicalGenerator.params if p.requires_grad)
    print("Total number of parameters in the quantum generator: ", total_params)

    data = ImageDataset(dataset_str)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1)  # 数据集
  
 
    #############################################################


    batches_done = 0
    
    weight_files = os.listdir(weight_dir)
    for i in weight_files:
        if 'generator' in i:
            weight_file = i
            break
    print(os.path.join(weight_dir, weight_file))
    generator.load_state_dict(torch.load(os.path.join(weight_dir, weight_file)))
  

    for epoch in trange(n_epochs):
        for i, batch in enumerate(dataloader):

            real_A = batch['A']

            real_B = batch['B']

            ############################生成器A2B的损失############################
            pred_images_B = generator(real_A)
        
            pred_images_A = generator(real_B)
    
      
            save_image(denorm(pred_images_B), os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
            save_image(denorm(real_A), os.path.join(out_dir_3, '{}.png'.format(batches_done)), nrow=5)

            
            # 保存wassterin以及图片
            save_image(denorm(pred_images_A), os.path.join(out_dir_2, '{}.png'.format(batches_done)), nrow=5)
            save_image(denorm(real_B), os.path.join(out_dir_4, '{}.png'.format(batches_done)), nrow=5)

         
            batches_done += 1
                
        
###############################
# if __name__ == "__main__":
#      train(r"E:\Acdamic\Data\DataMake\Mnist_C\Canny\Canny_7_shuffle", 10, 0)

