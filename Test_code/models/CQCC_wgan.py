import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalGAN1():
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, stride=2, kernel_size=4, padding=1),  # 28*28 -> 14*14
                nn.BatchNorm2d(16),
                nn.LeakyReLU()
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 16, stride=1, kernel_size=3, padding=1),  # 14*14 -> 14*14
                nn.BatchNorm2d(16),
                nn.LeakyReLU()
            )
            self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(16, 1, stride=2, kernel_size=4, padding=1),  # 14*14 -> 28*28
                nn.Tanh()
            )

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            return out


    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, stride=2, kernel_size=4, padding=1),  # 28*28 -> 14*14
                nn.BatchNorm2d(16),
                nn.LeakyReLU()
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, stride=2, kernel_size=4, padding=1),  # 14*14 -> 7*7
                nn.BatchNorm2d(32),
                nn.LeakyReLU()
            )
            self.fc = nn.Linear(7 * 7 * 32, 1)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out