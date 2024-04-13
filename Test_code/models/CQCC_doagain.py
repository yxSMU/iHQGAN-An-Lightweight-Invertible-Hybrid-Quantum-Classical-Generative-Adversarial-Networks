import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalGAN1():
    def __init__(self,  input_shape,output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    class ClassicalGenerator(nn.Module):
        def __init__(self, input_shape, output_shape):
            super().__init__()

            self.input_shape = input_shape  # 添加 input_shape 属性
            self.output_shape = output_shape  # 添加 output_shape 属性

            self.fc1 = nn.Linear(int(np.prod(input_shape)), 256)
            self.fc2 = nn.Linear(256, 512)
            self.fc3 = nn.Linear(512, 1024)
            self.fc4 = nn.Linear(1024, int(np.prod(output_shape)))

        def forward(self, x):
            x = x.view(x.shape[0], -1)  # 将输入 x 展平成一维向量
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.leaky_relu(self.fc3(x), 0.2)
            x = torch.tanh(self.fc4(x))
            x = x.view(x.shape[0], *self.output_shape)
            return x

    class ClassicalCritic(nn.Module):
        def __init__(self, image_shape):
            super().__init__()
            self.image_shape = image_shape

            self.fc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            return self.fc3(x)