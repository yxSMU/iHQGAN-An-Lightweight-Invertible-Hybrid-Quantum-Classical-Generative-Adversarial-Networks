import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

def add_noise(image, noise_level):
    """
    添加高斯噪声到图像
    """
    noise = torch.randn_like(image) * noise_level
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def load_noisy_mnist(file_location='./datasets', image_size=None, noise_level=0.1):
    if not image_size is None:
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    mnist_train = torchvision.datasets.MNIST(root=file_location, train=True, download=True, transform=transform)

    noisy_mnist_data = []
    for image, label in mnist_train:
        noisy_image = add_noise(image, noise_level)
        noisy_mnist_data.append((noisy_image, label))

    return noisy_mnist_data

# 示例用法
noise_level = 0.1  # 噪声水平
noisy_mnist_data = load_noisy_mnist(file_location='./datasets', image_size=None, noise_level=noise_level)