import os
import csv
import glob
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # 添加小常数 epsilon，避免除零或负数
    epsilon = 1e-6
    sigma1 += epsilon * np.eye(sigma1.shape[0])
    sigma2 += epsilon * np.eye(sigma2.shape[0])
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
def calculate_snr(original_image, denoised_image):
    # 计算噪声图像
    noise_image = original_image - denoised_image
    
    # 计算信号的均方根（RMS）
    signal_rms = np.sqrt(np.mean(original_image ** 2))
    
    # 计算噪声的均方根（RMS）
    noise_rms = np.sqrt(np.mean(noise_image ** 2))
    
    # 计算信号的功率
    signal_power = signal_rms ** 2
    
    # 计算噪声的功率
    noise_power = noise_rms ** 2
    
    # 计算信噪比（SNR）
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr

def Metric_Calculate(gen_dir, real_dir, metric_save):
    folder1_path = gen_dir
    folder2_path = real_dir
    num_images = 250

    print(folder1_path)
    print(folder2_path)
    # 获取文件夹中最后500张图片
    image_files_folder1 = glob.glob(os.path.join(folder1_path, '*.png'))
    image_files_folder1.sort(key=os.path.getmtime)
    last_500_images_folder1 = image_files_folder1[-num_images:]

    image_files_folder2 = glob.glob(os.path.join(folder2_path, '*.png'))
    image_files_folder2.sort(key=os.path.getmtime)
    last_500_images_folder2 = image_files_folder2[-num_images:]

    # 计算RMSE值并存储
    rmse_values = []
    frobenius_norm_values=[]
    fid_values= []
    psnr_value = []
    ssim_value=[]
    snr_value=[]
    for i in range(num_images):
        image_path_1 = last_500_images_folder1[i]
        image_path_2 = last_500_images_folder2[i]

        image1 = Image.open(image_path_1).convert("RGB")
        image2 = Image.open(image_path_2).convert("RGB")

        # 转换为单通道图像
        image1_gray = image1.convert("L")
        image2_gray = image2.convert("L")

        # image1_gray = Image.open(image_path_1).convert("L")
        # image2_gray = Image.open(image_path_2).convert("RGB").convert("L")

        image1_array = np.array(image1_gray)
        image2_array = np.array(image2_gray)

        fid = calculate_fid(image1_array, image2_array)
        log_fid=np.log(fid)
        fid_values.append(log_fid)

        # # 计算mse
        # rmse = np.sqrt(np.mean((image1_array - image2_array)**2))
        # rmse_values.append(rmse)

        # 计算差异矩阵
        diff_matrix = image1_array - image2_array
        # 计算Frobenius范数
        frobenius_norm = np.linalg.norm(diff_matrix, 'fro')
        log_frobenius_norm = np.log(frobenius_norm)
        frobenius_norm_values.append(log_frobenius_norm)

        # # 计算 SSIMmsssim
        # # 计算多尺度结构相似性
        # ms_ssim_value, _ = ssim(image1_array, image2_array, full=True, multichannel=True)
        # ssim_value.append(ms_ssim_value)

        # 计算PSNR
        psnr = peak_signal_noise_ratio(image1_array, image2_array)
        psnr_value.append(psnr)

        # 计算信噪比
        snr = calculate_snr(image1_array, image2_array)
        snr_value.append(snr)


    ###############################################################
    # 计算rmse
    # average_rmse = np.mean(rmse_values)
    # std_rmse = np.std(rmse_values)
    # # 计算fid
    average_fid = np.mean(fid_values)
    std_fid = np.std(fid_values)
    # # 计算SSiM
    # average_ssim = np.mean(ssim_value)
    # std_ssim = np.std(ssim_value)
    # 计算F-norm
    average_frobenius_norm = np.mean(frobenius_norm_values)
    std_from = np.std(frobenius_norm_values)
    # 计算PNSR平均值、最大值和最小值
    average_psnr_value= np.mean(psnr_value)
    std_pnsr = np.std(average_psnr_value)

    average_snr_value= np.mean(snr_value)
    std_snr = np.std(average_snr_value)
    

    # metric_result = [
    #     ["Metric", "Average"],
    #     # ["RMSE", f"{average_rmse:.2f}+-{std_rmse:.2f}"],
    #     ["FID", f"{average_fid:.2f}+-{std_fid:.2f}"],
    #     # ["SSIM", f"{average_ssim:.2f}+-{std_ssim:.2f}"],
    #     ["FORM", f"{average_frobenius_norm:.2f}+-{std_from:.2f}"],
    #     # ["PSNR", f"{average_psnr_value:.2f}+-{std_pnsr:.2f}"],
    #     ["SNR", f"{average_snr_value:.2f}+-{std_snr:.2f}"]
    # ]

    metric_result = [
        ["Metric", "Average"],
        # ["RMSE", f"{average_rmse:.2f}+-{std_rmse:.2f}"],
        ["FID", f"{average_fid:.2f}"],
        # ["SSIM", f"{average_ssim:.2f}+-{std_ssim:.2f}"],
        ["FORM", f"{average_frobenius_norm:.2f}"],
        # ["PSNR", f"{average_psnr_value:.2f}+-{std_pnsr:.2f}"],
        ["SNR", f"{average_snr_value:.2f}"]
    ]


    csv_file_path = metric_save
    with open(csv_file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(metric_result)

    print(f"Data saved to {csv_file_path}")


    # # print(f"RMSE范围: {average_rmse:.2f} +{std_rmse:.2f}/-{std_rmse:.2f} 小好")
    # print(f"FID范围: {average_fid:.2f} +{std_fid:.2f}/-{std_fid:.2f} 小好")
    # # print(f"SSIM范围: {average_ssim:.2f} +{std_ssim:.2f}/-{std_ssim:.2f} 大好")
    # print(f"FORM范围: {average_frobenius_norm:.2f} +{std_from:.2f}/-{std_from:.2f} 小好")
    # # print(f"PSNR范围: {average_psnr_value:.2f} +{std_pnsr:.2f}/-{std_pnsr:.2f} 大好")
    # print(f"SNR范围: {average_snr_value:.2f} +{std_pnsr:.2f}/-{std_snr:.2f} 大好")

       # print(f"RMSE范围: {average_rmse:.2f} +{std_rmse:.2f}/-{std_rmse:.2f} 小好")
    print(f"FID范围: {average_fid:.2f}  小好")
    # print(f"SSIM范围: {average_ssim:.2f} +{std_ssim:.2f}/-{std_ssim:.2f} 大好")
    print(f"FORM范围: {average_frobenius_norm:.2f}小好")
    # print(f"PSNR范围: {average_psnr_value:.2f} +{std_pnsr:.2f}/-{std_pnsr:.2f} 大好")
    print(f"SNR范围: {average_snr_value:.2f} 大好")