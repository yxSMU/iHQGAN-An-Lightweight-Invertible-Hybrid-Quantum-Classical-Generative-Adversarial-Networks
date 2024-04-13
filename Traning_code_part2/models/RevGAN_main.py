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

def UnpairedRevGANModel(dataset_str):
  
    netG_A_enc = define_G_enc(1, 1, 64, "resnet_9blocks", "batch", 'normal', 0.02, 2)

    netG_core = define_G_core(1, 1, 64, "resnet_9blocks", "batch",True,'normal', 0.02, True, 2, 'additive')

    netG_A_dec =define_G_dec(1, 1, 64, "resnet_9blocks", "batch",'normal', 0.02,  not True, 2)

    netG_B_enc = define_G_enc(1, 1, 64, "resnet_9blocks", "batch",  'normal', 0.02,  2)
    netG_B_dec = define_G_dec(1, 1, 64, "resnet_9blocks", "batch", 'normal', 0.02,  not True, 2)

    netD_A = define_D(1, 64, "basic",3, "batch", True, 'normal', 0.02)
    netD_B = define_D(1, 64, "basic", 3, "batch", True, 'normal', 0.02 )

    total_params1 = sum(p.numel() for p in netG_A_enc.parameters() if p.requires_grad)
    total_params2 = sum(p.numel() for p in netG_core.parameters() if p.requires_grad)
    total_params3 = sum(p.numel() for p in netG_A_dec.parameters() if p.requires_grad)
    total_params4 = sum(p.numel() for p in netG_B_enc.parameters() if p.requires_grad)
    total_params5 = sum(p.numel() for p in netG_B_dec.parameters() if p.requires_grad)
    total_params=total_params1+total_params2+total_params3+total_params4+total_params5

    # total_params = sum(p.numel() for p in gan_1.ClassicalGenerator.params if p.requires_grad)
    print("Total number of parameters in the quantum generator: ", total_params)
  
    data = ImageDataset(dataset_str)
    dataloader = DataLoader(data, batch_size=10, shuffle=False, num_workers=1)  # 数据集
   
     #####输出的文件路径##########
  
    out_dir = "E:\Paper\Data\TrainResult\REVgan\REVgan_canny7\\result_A2B"
    out_dir_2 = "E:\Paper\Data\TrainResult\REVgan\REVgan_canny7\\result_B2A"
    out_dir_3 = "E:\Paper\Data\TrainResult\REVgan\REVgan_canny7\\result_A_real"
    out_dir_4 = "E:\Paper\Data\TrainResult\REVgan\REVgan_canny7\\result_B_real"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_2, exist_ok=True)
    os.makedirs(out_dir_3, exist_ok=True)
    os.makedirs(out_dir_4, exist_ok=True)
 
 
    # define loss functions
    criterionGAN = GANLoss(use_lsgan=not True)
    criterionCycle = torch.nn.L1Loss()
    criterionIdt = torch.nn.L1Loss()
    # initialize optimizers
    params_G = [netG_A_enc.parameters(),
                            netG_core.parameters(),
                            netG_A_dec.parameters(),
                            netG_B_enc.parameters(),
                            netG_B_dec.parameters()]
    optimizer_G = torch.optim.Adam(itertools.chain(*params_G
                                            ),
                                            lr=0.0002, betas=(0.5, 0.999))
 
    params_D_A = netD_A.parameters()
    optimizer_D_A = torch.optim.Adam(params_D_A, lr=0.0002, betas=(0.5, 0.999))

    params_D_B = netD_B.parameters()
    optimizer_D_B = torch.optim.Adam(params_D_B, lr=0.0002, betas=(0.5, 0.999))

    lambda_idt = 0.5
    lambda_A = 10
    lambda_B = 10
    
    n_epochs=20
    # 使用 nn.Conv2d 进行通道数增加
    batches_done=0
    
    loss_criticA_history = []  # wasserstein距离
    loss_criticB_history = []  # wasserstein距离

    loss1_history = []  # wasserstein距离
    loss2_history = []  # wasserstein距离

    for epoch in trange(n_epochs):
      for i, batch in enumerate(dataloader):

        real_A = batch['A']

        real_B = batch['B']
       
        
        # Forward cycle (A to B)
        fake_B = netG_A_dec(netG_core(netG_A_enc(real_A)))
        rec_A = netG_B_dec(netG_core(netG_B_enc(fake_B), inverse=True))

        # Backward cycle (B to A)
        fake_A = netG_B_dec(netG_core(netG_B_enc(real_B), inverse=True))
        rec_B = netG_A_dec(netG_core(netG_A_enc(fake_A)))
  
        # Identity loss
        # G_A should be identity if real_B is fed.
        idt_A = netG_A_dec(netG_core(netG_A_enc(real_B)))
       
        loss_idt_A = criterionIdt(idt_A, real_B) * lambda_B * lambda_idt
      
      
        # G_B should be identity if real_A is fed.
        idt_B = netG_B_dec(netG_core(netG_B_enc(real_A), inverse=True))
        loss_idt_B = criterionIdt(idt_B, real_A) * lambda_A * lambda_idt
        
        # GAN loss D_A(G_A(A))
        loss_G_A = criterionGAN(netD_A(fake_B), True)
        # GAN loss D_B(G_B(B))
        loss_G_B = criterionGAN(netD_B(fake_A), True)
        # Forward cycle loss
        loss_cycle_A = criterionCycle(rec_A, real_A) * lambda_A
        # Backward cycle loss
        loss_cycle_B = criterionCycle(rec_B, real_B) * lambda_B

        gloss_a2b=loss_G_A+loss_cycle_A+loss_idt_A
        gloss_b2a=loss_G_B+loss_cycle_B+loss_idt_B
       

        loss1_history.append(gloss_a2b.item())
        loss2_history.append(gloss_b2a.item())
        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        optimizer_G.zero_grad()
           
        # a2b 生成b 
        # 将张量传递给卷积操作，得到通道数为 256 的输出张量
        pred_real_B = netD_B(real_B)
        loss_D_real_B = criterionGAN(pred_real_B, True)
        # Fake
        pred_fake_B = netD_B(fake_B)
        loss_D_fake_B = criterionGAN(pred_fake_B, False)
        # Combined loss
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_criticA_history.append(loss_D_B.item())
        optimizer_D_B.zero_grad()

        # b2a 生成a
        # Real
        pred_real_A = netD_A(real_A)
        loss_D_real_A = criterionGAN(pred_real_A, True)
        # Fake
        pred_fake_A = netD_A(fake_A)
        loss_D_fake_A = criterionGAN(pred_fake_A, False)
        # Combined loss
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_criticB_history.append(loss_D_A.item())
        optimizer_D_A.zero_grad()
      
        loss_G.backward(retain_graph=True)
        loss_D_B.backward(retain_graph=True)
        loss_D_A.backward()
        
        optimizer_G.step()
        optimizer_D_B.step()
        optimizer_D_A.step()
        
        save_image(denorm(fake_B), os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
        save_image(denorm(real_A), os.path.join(out_dir_3, '{}.png'.format(batches_done)), nrow=5)
   
        save_image(denorm(fake_A), os.path.join(out_dir_2, '{}.png'.format(batches_done)), nrow=5)
        save_image(denorm(real_B), os.path.join(out_dir_4, '{}.png'.format(batches_done)), nrow=5)
        
        np.save(os.path.join(out_dir, 'loss_criticA_history.npy'), loss_criticA_history)
        np.save(os.path.join(out_dir, 'loss_criticB_history.npy'), loss_criticB_history)

        np.save(os.path.join(out_dir, 'loss_1_history.npy'), loss1_history)
        np.save(os.path.join(out_dir, 'loss_2_history.npy'), loss2_history)


        batches_done=batches_done+1
        
  
  
    torch.save(netG_A_enc.state_dict(), os.path.join(out_dir, 'netG_A_enc-{}.pt'.format(batches_done)))
    torch.save(netG_core.state_dict(), os.path.join(out_dir, 'netG_core-{}.pt'.format(batches_done)))
    torch.save(netG_A_dec.state_dict(), os.path.join(out_dir, 'netG_A_dec-{}.pt'.format(batches_done)))
    torch.save(netG_B_enc.state_dict(), os.path.join(out_dir, 'netG_B_enc-{}.pt'.format(batches_done)))
    torch.save(netG_B_dec.state_dict(), os.path.join(out_dir, 'netG_B_dec-{}.pt'.format(batches_done)))
    print("Finish!")
if __name__ == "__main__":
    UnpairedRevGANModel(r"E:\Acdamic\Data\DataMake\Mnist_C\Canny\Canny_7_shuffle")

 
#REVgan_Dilate1