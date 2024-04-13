import torch.nn as nn
import torch.nn.functional as F
import torch

# 定义降采样部分
class downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsample, self).__init__()
        self.down = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.down(x)


# 定义上采样部分
class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=False):
        super(upsample, self).__init__()
        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5) if drop_out else nn.Identity()
        )

    def forward(self, x):
        return self.up(x)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class ClassicalGAN1():
    def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
        """Custom deconvolutional layer for simplicity."""
        layers = []
        layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
        if bn:
            layers.append(nn.BatchNorm2d(c_out))
        return nn.Sequential(*layers)


    def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
        """Custom convolutional layer for simplicity."""
        layers = []
        layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
        if bn:
            layers.append(nn.BatchNorm2d(c_out))
        return nn.Sequential(*layers)

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()

            self.down_1 = nn.Conv2d(1, 64, 4, 2, 1)  # [batch, 1, 32, 32] => [batch, 64, 16, 16]

            for i in range(7):
                if i == 0:
                    self.down_2 = downsample(64, 128)  # [batch, 64, 16, 16] => [batch, 128, 8, 8]
                    self.down_3 = downsample(128, 256)  # [batch, 128, 8, 8] => [batch, 256, 4, 4]

            for i in range(7):
                if i == 0:
                    self.up_1 = upsample(256, 128)  # [batch, 256, 4, 4] => [batch, 128, 8, 8]
                    self.up_2 = upsample(256, 64, drop_out=True)  # [batch, 256, 8, 8] => [batch, 64, 16, 16]

            self.last_Conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )

            self.init_weight()

        def init_weight(self):
            for w in self.modules():
                if isinstance(w, nn.Conv2d):
                    nn.init.kaiming_normal_(w.weight, mode='fan_out')
                    if w.bias is not None:
                        nn.init.zeros_(w.bias)
                elif isinstance(w, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(w.weight, mode='fan_in')
                elif isinstance(w, nn.BatchNorm2d):
                    nn.init.ones_(w.weight)
                    nn.init.zeros_(w.bias)

        def forward(self, x):
            down_1 = self.down_1(x)
            down_2 = self.down_2(down_1)
            down_3 = self.down_3(down_2)

            up_1 = self.up_1(down_3)
            up_2 = self.up_2(torch.cat([up_1, down_2], dim=1))
            out = self.last_Conv(torch.cat([up_2, down_1], dim=1))

            return out



    class Discriminator(nn.Module):
        """Discriminator for mnist."""

        def __init__(self, conv_dim=16):
            super().__init__()

            self.conv1 = conv(1, conv_dim, 4, bn=False)
            self.conv2 = conv(conv_dim, conv_dim * 2, 4)
            self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
            self.fc = conv(conv_dim * 4, 1, 4, 1, 0, False)

        def forward(self, x):
            out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 16, 16)
            out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
            out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
            out = self.fc(out).squeeze()
            return out


