
import torch
import torch.nn as nn
from models.unet import UNet

# PatchGAN
class PatchGAN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)
