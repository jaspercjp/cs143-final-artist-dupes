import torch
import torch.nn as nn
from torch import Tensor


# architeecture of residual blocks taken from 
# AUTHOR: https://medium.com/@chen-yu/building-a-customized-residual-cnn-with-pytorch-471810e894ed 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding='same',
        )

        self.batch_norm1 = nn.batchNorm2(out_channels)

        self.leaky_relu = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding='same',
        )

        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = None

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels = out_channels,
                    kernel_size = (3, 3),
                    padding = 'same'
                ),
                nn.BatchNorm2d(out_channels = out_channels)
            )

        
        # condensed into a single block, lacking the final addition
        self.block = nn.Sequential(
            self.conv1, 
            self.batch_norm1, 
            self.leaky_relu, 
            self.conv2,
            self.batch_norm2, 
            #lacks the final addition to the identity
        )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        x = self.block(x)

        return self.leaky_relu(x + identity)

