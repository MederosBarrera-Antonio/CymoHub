
# U-NET MODEL (BASIC AND WITH ATTENTION GATES)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi.expand_as(x)

class UNet(nn.Module):
    # Basic U-Net architectura based on: https://github.com/milesial/Pytorch-UNet/tree/master

    def __init__(self, n_channels, n_classes, num_filters=64, bilinear=False, use_attention_gates=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention_gates = use_attention_gates
        
        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(n_channels, num_filters)
        self.down1 = Down(num_filters, num_filters*2)
        self.down2 = Down(num_filters*2, num_filters*4)
        self.down3 = Down(num_filters*4, num_filters*8 // factor)
        self.down4 = Down(num_filters*8, num_filters*16 // factor)
        
        self.up1 = Up(num_filters*16, num_filters*8 // factor, bilinear)
        self.up2 = Up(num_filters*8, num_filters*4 // factor, bilinear)
        self.up3 = Up(num_filters*4, num_filters*2 // factor, bilinear)
        self.up4 = Up(num_filters*2, num_filters, bilinear)
        self.outc = OutConv(num_filters, n_classes)

        if use_attention_gates:
            self.att1 = AttentionGate(F_g=num_filters*16 // factor, 
                                      F_l=num_filters*8 // factor,
                                      F_int=num_filters*8 // (2*factor))
            self.att2 = AttentionGate(F_g=num_filters*8 // factor,
                                      F_l=num_filters*4 // factor,
                                      F_int=num_filters*4 // (2*factor))
            self.att3 = AttentionGate(F_g=num_filters*4 // factor,
                                      F_l=num_filters*2 // factor,
                                      F_int=num_filters*2 // (2*factor))
            self.att4 = AttentionGate(F_g=num_filters*2,
                                      F_l=num_filters,
                                      F_int=num_filters // (2*factor))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        if self.use_attention_gates:
            x3 = self.att2(g=x4, x=x3)
        x = self.up2(x4, x3)
        if self.use_attention_gates:
            x2 = self.att3(g=x, x=x2)
        x = self.up3(x, x2)
        if self.use_attention_gates:
            x1 = self.att4(g=x, x=x1)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits