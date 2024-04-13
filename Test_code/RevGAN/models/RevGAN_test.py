import torch
import itertools
import os
import torch.nn as nn
# from .base_model import BaseModel
from networks import define_G_enc,define_G_core,define_G_dec,define_D,GANLoss
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from UntilClassical.datasets_2_padding import ImageDataset
from tqdm import trange
import numpy as np
from torchvision.utils import save_image
from UntilClassical.dataset import  denorm

def UnpairedRevGANModel(dataset_str, weight_dir, result_dir):
    netG_A_enc = define_G_enc(1, 1, 64, "resnet_9blocks", "batch", 'normal', 0.02, 2)

    netG_core = define_G_core(1, 1, 64, "resnet_9blocks", "batch",True,'normal', 0.02, True, 2, 'additive')

    netG_A_dec =define_G_dec(1, 1, 64, "resnet_9blocks", "batch",'normal', 0.02,  not True, 2)

    netG_B_enc = define_G_enc(1, 1, 64, "resnet_9blocks", "batch",  'normal', 0.02,  2)
    netG_B_dec = define_G_dec(1, 1, 64, "resnet_9blocks", "batch", 'normal', 0.02,  not True, 2)

 

  
    data = ImageDataset(dataset_str)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)  # 数据集
   
     #####输出的文件路径##########
  
    out_dir = result_dir + "/result_A2B"
    out_dir_2 = result_dir + "/result_B2A"
    out_dir_3 = result_dir + "/result_A_real"
    out_dir_4 = result_dir + "/result_B_real"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_2, exist_ok=True)
    os.makedirs(out_dir_3, exist_ok=True)
    os.makedirs(out_dir_4, exist_ok=True)
   

    weight_files = os.listdir(weight_dir)
    for i in weight_files:
        if 'netG_A_enc' in i:
          weight_1 = i
        elif 'netG_core' in i:
          weight_2 = i
        elif 'netG_A_dec' in i:
          weight_3 = i
        elif 'netG_B_enc' in i:
          weight_4 = i
        elif 'netG_B_dec' in i:
          weight_5 = i
        else:
          continue
    netG_A_enc.load_state_dict(torch.load(os.path.join(weight_dir, weight_1)))
    netG_core.load_state_dict(torch.load(os.path.join(weight_dir, weight_2)))
    netG_A_dec.load_state_dict(torch.load(os.path.join(weight_dir, weight_3)))
    netG_B_enc.load_state_dict(torch.load(os.path.join(weight_dir, weight_4)))
    netG_B_dec.load_state_dict(torch.load(os.path.join(weight_dir, weight_5)))

  
   
    n_epochs = 1
    # 使用 nn.Conv2d 进行通道数增加

    
    batches_done = 0
    for epoch in trange(n_epochs):
      for i, batch in enumerate(dataloader):

        real_A = batch['A']

        real_B = batch['B']
       
        
        # Forward cycle (A to B)
        fake_B = netG_A_dec(netG_core(netG_A_enc(real_A)))
       
        # Backward cycle (B to A)
        fake_A = netG_B_dec(netG_core(netG_B_enc(real_B), inverse=True))

        save_image(denorm(fake_B), os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
        save_image(denorm(real_A), os.path.join(out_dir_3, '{}.png'.format(batches_done)), nrow=5)
   
        save_image(denorm(fake_A), os.path.join(out_dir_2, '{}.png'.format(batches_done)), nrow=5)
        save_image(denorm(real_B), os.path.join(out_dir_4, '{}.png'.format(batches_done)), nrow=5)
        
        batches_done=batches_done+1
      
        print(1)
  
    
if __name__ == "__main__":
    data_basepath = "E:\ACMMM\YJC\YJC\Data"
    weight_basepath = "E:\ACMMM\YJC\YJC\Weight_models"
    result_basepath = "E:\ACMMM\YJC\YJC\TestResult"
    
    # data_basepath = "../../../Data"
    # weight_basepath = "../../../Weight_models"
    # result_basepath = "../../../TestResult"
    code = 'REVgan'
    # metric_basepath = "../../../Metric_Result"
    # for data in ['Canny', 'Dilate', 'Noise']:
    for data in ['Canny', 'Dilate', 'Noise']:
        for number in ['0','1','2','3','4','5','6','7']:
            data_path = os.path.join(data_basepath, data, number)
            weight_path = os.path.join(weight_basepath, code, data+number)
            result_path = os.path.join(result_basepath, code+'_test', data, number)
            # metric_path = os.path.join(metric_basepath, code+'_test')
            print("\ndata path: "+data_path)
            print("weight path: "+weight_path)
            print("result path: "+result_path)
            UnpairedRevGANModel(data_path, weight_path, result_path)

 
