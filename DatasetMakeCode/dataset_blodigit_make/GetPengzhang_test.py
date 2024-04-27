import torch
from torchvision import datasets, transforms
import cv2
from torchvision.utils import save_image
import numpy as np
import os

torch.manual_seed(42)
# 定义 MNIST 数据集和加载器
transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# dataloader_mnist = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=True)
# 下载MNIST数据集
# 筛选出类别为1的前100个样本
filtered_indices = [i for i in range(len(mnist_dataset)) if mnist_dataset.targets[i] == 2][1000:1250]
# 创建两个文件夹用于保存图像
os.makedirs(r'E:\ACMMM\YJC\YJC\Data\Dilate\2\\A', exist_ok=True)
os.makedirs(r'E:\ACMMM\YJC\YJC\Data\Dilate\2\\B', exist_ok=True)

# 定义膨胀核
kernel = np.ones((2, 2), np.uint8)
for i in filtered_indices:
    image, label = mnist_dataset[i]
# 对 MNIST 数字图像进行膨胀处理
    # 将图像转为 NumPy 数组
    image_np = (image.squeeze().numpy() * 255).astype(np.uint8)  # 将图像转为 uint8 格式
    # 对图像进行膨胀处理

    torch_image = torch.from_numpy(image_np / 255.0).unsqueeze(0)
    # 将处理后的图像保存
    original_path = 'J:\\dataset\\B\\origin_{}.png'.format(i)
    save_image(torch_image, original_path)

    # 将处理后的图像保存
    dilated_image = cv2.dilate(image_np, kernel, iterations=1)
    torch_image_binarized=torch.from_numpy(dilated_image / 255.0).unsqueeze(0)
    dilate_path  = 'E:\ACMMM\YJC\YJC\Data\Dilate\2\\A\\dilate{}.png'.format(i)
    save_image(torch_image_binarized ,dilate_path)

    print(i)
    # if i == 4:  # 保存5张图像作为示例
    #     break
