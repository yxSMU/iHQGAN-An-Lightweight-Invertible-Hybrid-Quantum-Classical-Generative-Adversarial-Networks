# import cv2
# # 读取单通道（灰度）图像
# gray_image = cv2.imread("clip\B\gray_130.png", cv2.IMREAD_GRAYSCALE)
# # 将图像的第一行像素值全部置为黑色
# gray_image[:11, :] = 0  # 0 表示黑色
# gray_image[29:45, :] = 0  # 0 表示黑色
# # 获取图像高度
# height = gray_image.shape[0]
# # 将图像的后4行像素值全部置为黑色
# gray_image[height-7:, :] = 0  # 0 表示黑色
# # 保存更改后的图像
# cv2.imwrite("modified_gray_image2.jpg", gray_image)

import os
import cv2

def process_and_save_images(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') ]
    for image_file in image_files:
        # 构造输入图像的完整路径
        input_path = os.path.join(input_folder, image_file)
        # 读取图像并转换为灰度图
        gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        # 修改图像像素值
        gray_image[:8, :] = 0
        gray_image[26:32, :] = 0
        # height = gray_image.shape[0]
        # gray_image[height-7:, :] = 0
        # 构造输出图像的完整路径
        output_path = os.path.join(output_folder, f"modified_{image_file}")
        # 保存更改后的图像
        cv2.imwrite(output_path, gray_image)
        print(f"Processed and saved: {output_path}")

# 调用函数处理整个文件夹下的图像

# file_index = [0, 1]
# for i in file_index:
#     data_type = "Canny/" + str(i)
#     input_folder = "../TestResult/QRGAN_test/" + data_type + "/result_A2B"
#     output_folder = "../TestDenoise/" + data_type + "/A2B_denoise"
#     input_folder2 = "../TestResult/QRGAN_test/" + data_type + "/result_B2A"
#     output_folder2 = "../TestDenoise/" + data_type + "/B2A_denoise"
#     process_and_save_images(input_folder, output_folder)
#     process_and_save_images(input_folder2, output_folder2)
