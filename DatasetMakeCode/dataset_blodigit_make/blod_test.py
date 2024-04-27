import torch
from torchvision import datasets, transforms
import cv2
from torchvision.utils import save_image
import numpy as np
import os

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

filtered_indices = [i for i in range(len(mnist_dataset)) if mnist_dataset.targets[i] == 2][1000:1250]

os.makedirs(r'E:\ACMMM\YJC\YJC\Data\Dilate\2\\A', exist_ok=True)
os.makedirs(r'E:\ACMMM\YJC\YJC\Data\Dilate\2\\B', exist_ok=True)


kernel = np.ones((2, 2), np.uint8)
for i in filtered_indices:
    image, label = mnist_dataset[i]

    image_np = (image.squeeze().numpy() * 255).astype(np.uint8) 
  

    torch_image = torch.from_numpy(image_np / 255.0).unsqueeze(0)
  
    original_path = 'J:\\dataset\\B\\origin_{}.png'.format(i)
    save_image(torch_image, original_path)

  
    dilated_image = cv2.dilate(image_np, kernel, iterations=1)
    torch_image_binarized=torch.from_numpy(dilated_image / 255.0).unsqueeze(0)
    dilate_path  = 'E:\ACMMM\YJC\YJC\Data\Dilate\2\\A\\dilate{}.png'.format(i)
    save_image(torch_image_binarized ,dilate_path)

    print(i)
 
   
