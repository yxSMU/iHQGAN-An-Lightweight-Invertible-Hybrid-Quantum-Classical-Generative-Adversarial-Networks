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

