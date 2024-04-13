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

from UntilClassical.dataset import load_mnist, load_fmnist, denorm, select_from_dataset
from UntilClassical.datasets_2_padding import ImageDataset
from UntilClassical.wgan import compute_gradient_penalty

from models.CQCC_wganRes import ClassicalGAN1
from models.CQCC_wganRes2 import ClassicalGAN2

from tqdm import trange


def train(dataset_str, batch_size,  checkpoint):

    device = torch.device('cpu')
    n_epochs = 20
    image_size = 32

    #
    # lr_D = 0.0002
    # lr_G = 0.000001

    lr_D = 0.0002
    lr_G = 0.0001

    # lr_D = 0.0001
    # lr_G = 0.00008

    b1 = 0
    b2 = 0.9

    lambda_gp = 10
    n_critic =2
    #####输出的文件路径##########

     # Create output dirs if they don't exist
  
    out_dir = r"E:\Paper\Data\TrainResult\ClasscisalTrain\ONE2ONE\Noise\ONE2ONE_Noise6\result_A2B"
    out_dir_2 = r"E:\Paper\Data\TrainResult\ClasscisalTrain\ONE2ONE\Noise\ONE2ONE_Noise6\result_B2A"
    out_dir_3 = r"E:\Paper\Data\TrainResult\ClasscisalTrain\ONE2ONE\Noise\ONE2ONE_Noise6\result_A_real"
    out_dir_4 =r"E:\Paper\Data\TrainResult\ClasscisalTrain\ONE2ONE\Noise\ONE2ONE_Noise6\result_B_real"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_2, exist_ok=True)
    os.makedirs(out_dir_3, exist_ok=True)
    os.makedirs(out_dir_4, exist_ok=True)

########################################################
    channels=1
    input_shape = (channels, image_size, image_size)
    output_shape = (channels, image_size, image_size)

    gan_1 = ClassicalGAN1()
    generator = gan_1.Generator()
    critic = gan_1.Discriminator()

    gan_2 = ClassicalGAN2()
    generator_2 = gan_2.Generator()
    critic_2 = gan_2.Discriminator()

    # 计算生成器的参数数量
    total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    # total_params = sum(p.numel() for p in gan_1.ClassicalGenerator.params if p.requires_grad)
    print("Total number of parameters in the quantum generator: ", total_params)

    data = ImageDataset(dataset_str)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1)  # 数据集
  

    optimizer_G = torch.optim.Adam(itertools.chain(generator.parameters()), lr=lr_G,
                                   betas=(b1, b2))

    optimizer_D_1 = Adam(critic.parameters(), lr=lr_D, betas=(b1, b2))  # 优化器
    optimizer_D_2 = Adam(critic_2.parameters(), lr=lr_D, betas=(b1, b2))  # 优化器

    
    

    wasserstein_distance_history = []  # wasserstein距离
    wasserstein_distance_history2 = []  # wasserstein距离

    loss_criticA_history = []  # wasserstein距离
    loss_criticB_history = []  # wasserstein距离

    loss1_history = []  # wasserstein距离
    loss2_history = []  # wasserstein距离

    ######################记录的数组####################################
    lossD1sum100 = 0
    lossD2sum100 = 0

    lossG1sum50 = 0
    lossG2sum50 = 0

    lossW1sum50 = 0
    lossW2sum50 = 0

    aver_loss_criticA_history = []
    aver_loss_criticB_history = []

    aver_loss1_history = []
    aver_loss2_history = []

    aver_wasserstein_distance_history = []
    aver_wasserstein_distance_history2 = []
    #############################################################

    batches_done = 0
    criterion_cycle = torch.nn.L1Loss()

    if checkpoint != 0:
        critic.load_state_dict(torch.load(out_dir + f"/critic-{checkpoint}.pt"))
        generator.load_state_dict(torch.load(out_dir + f"/generator-{checkpoint}.pt"))

        critic_2.load_state_dict(torch.load(out_dir + f"/critic_2-{checkpoint}.pt"))
        

        wasserstein_distance_history = list(np.load(out_dir + "/wasserstein_distance.npy"))
        wasserstein_distance_history2 = list(np.load(out_dir + "/wasserstein_distance2.npy"))


    for epoch in trange(n_epochs):
        for i, batch in enumerate(dataloader):

            real_A = batch['A']

            real_B = batch['B']


            optimizer_G.zero_grad()
            real_A = real_A.to(device)
            ############################生成器A2B的损失############################
            pred_images_B = generator(real_A)
            fake_validity_B = critic(pred_images_B)
            g_loss_1 = -torch.mean(fake_validity_B)

            recover_A = generator(pred_images_B)
            g_loss_cycle_1 = criterion_cycle(real_A, recover_A) * 20

            gloss_a2b = g_loss_1 + g_loss_cycle_1
            loss1_history.append(gloss_a2b.item())
            gloss_a2b.backward()
            optimizer_G.step()


            ############################生成器B2A的损失############################
            
            optimizer_G.zero_grad()
            pred_images_A = generator(real_B)
            fake_validity_A = critic_2(pred_images_A)
            g_loss_2 = -torch.mean(fake_validity_A)

            recover_B = generator(pred_images_A)
            g_loss_cycle_2 = criterion_cycle(real_B, recover_B) *20
            gloss_b2a = g_loss_2 + g_loss_cycle_2 
            loss2_history.append(gloss_b2a.item())
            gloss_b2a.backward()
            optimizer_G.step()
            ############################生成器cycleABA的损失############################

            save_image(denorm(pred_images_B), os.path.join(out_dir, '{}.png'.format(batches_done)), nrow=5)
            save_image(denorm(real_A), os.path.join(out_dir_3, '{}.png'.format(batches_done)), nrow=5)
            torch.save(generator.state_dict(), os.path.join(out_dir, 'generator-{}.pt'.format(batches_done)))
            print("saved images and state")

            ######################生成器cycleABA的损失#################################
            
            # 保存wassterin以及图片
            save_image(denorm(pred_images_A), os.path.join(out_dir_2, '{}.png'.format(batches_done)), nrow=5)
            save_image(denorm(real_B), os.path.join(out_dir_4, '{}.png'.format(batches_done)), nrow=5)


            np.save(os.path.join(out_dir, 'wasserstein_distance.npy'), wasserstein_distance_history)
            np.save(os.path.join(out_dir, 'wasserstein_distance2.npy'), wasserstein_distance_history2)
            
            batches_done += 1
                
            optimizer_D_1.zero_grad()
            ######################输入A 生成假的B#######################
            fake_images_B = generator(real_A)
            ######################判别真的B 假的B 更新判别器（1）############################
            real_validity_B = critic(real_B)

            fake_validity_B = critic(fake_images_B)

            #########################判别器1计算损失loss##################################
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_B, fake_images_B, device)
            # Adversarial loss
            loss_critic_A = -torch.mean(real_validity_B) + torch.mean(fake_validity_B) + lambda_gp * gradient_penalty
            loss_criticA_history.append(loss_critic_A.item())
            lossD1sum100 = lossD1sum100 + loss_critic_A.item()
            # Adversarial loss
            loss_critic_A.backward(retain_graph=True)

            optimizer_D_1.step()

            #######################更新判别器2#############################################
            optimizer_D_2.zero_grad()
            ######################输入B 生成假的A##################################
            fake_images_A =generator(real_B)
            fake_images_A = fake_images_A.to(device)

            ######################判别真的A 假的A 更新判别器（2）############################
            real_validity_A = critic_2(real_A)
            real_validity_A = real_validity_A.to(device)

            fake_validity_A = critic_2(fake_images_A)
            fake_validity_A = fake_validity_A.to(device)
            #########################判别器2计算损失loss##################################
            # Gradient penalty
            gradient_penalty_2 = compute_gradient_penalty(critic_2, real_A, fake_images_A, device)
            # Adversarial loss
            loss_critic_B = -torch.mean(real_validity_A) + torch.mean(fake_validity_A) + lambda_gp * gradient_penalty_2
            loss_criticB_history.append(loss_critic_B.item())

            
            lossD2sum100 = lossD2sum100 + loss_critic_B.item()
            loss_critic_B.backward(retain_graph=True)

            optimizer_D_2.step()
            print(
                f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D1 loss: {loss_critic_A.item()}] [D2 loss: {loss_critic_B.item()}] [G1 loss: {g_loss_1.item()}][G2 loss: {g_loss_2.item()}] ")

            np.save(os.path.join(out_dir, 'loss_criticA_history.npy'), loss_criticA_history)
            np.save(os.path.join(out_dir, 'loss_criticB_history.npy'), loss_criticB_history)
            
 ###########################每个epoch记录#######################
        aver_loss_criticA = lossD1sum100 / 100
        aver_loss_criticB = lossD2sum100 / 100

        aver_loss1 = lossG1sum50 / 50
        aver_loss2 = lossG2sum50 / 50

        print(
            f"[Epoch {epoch}/{n_epochs}][D1 loss: {aver_loss_criticA}] [D2 loss: {aver_loss_criticB}] [G1 loss: {aver_loss1}][G2 loss: {aver_loss2}] ")

        aver_loss_criticA_history.append(aver_loss_criticA)
        aver_loss_criticB_history.append(aver_loss_criticB)

        aver_loss1_history.append(aver_loss1)
        aver_loss2_history.append(aver_loss2)

 

        np.save(os.path.join(out_dir, 'aver_loss_criticA_history.npy'), aver_loss_criticA_history)
        np.save(os.path.join(out_dir, 'aver_loss_criticB_history.npy'), aver_loss_criticB_history)

        np.save(os.path.join(out_dir, 'aver_loss1_history.npy'), aver_loss1_history)
        np.save(os.path.join(out_dir, 'aver_loss2_history.npy'), aver_loss2_history)

        np.save(os.path.join(out_dir, 'aver_wasserstein_distance_history.npy'), aver_wasserstein_distance_history)
        np.save(os.path.join(out_dir, 'aver_wasserstein_distance_history2.npy'), aver_wasserstein_distance_history2)

        lossD1sum100 = 0
        lossD2sum100 = 0

        lossG1sum50 = 0
        lossG2sum50 = 0

        lossW1sum50 = 0
        lossW2sum50 = 0

###############################
if __name__ == "__main__":
     train(r"E:\Acdamic\Data\DataMake\MySelfData\Noise\Noise_6_shuffle", 10, 0)

