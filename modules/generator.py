import torch
import torch.nn as nn

from .components import *

class Block(nn.Module):
    def __init__(self, input_channels, out_channels, num_classes):
        super(Block, self).__init__()
        self.chbn_0 = ConditionalBatchNorm(num_classes)
        self.chbn_1 = ConditionalBatchNorm(num_classes)

        self.snconv1 = SNConv2d(input_channels, out_channels, 3, 1, 1)
        self.snconv2 = SNConv2d(out_channels, out_channels, 3, 1, 1)
        self.snconv3 = SNConv2d(input_channels, out_channels, 1, 1, 1)

        self.usample = nn.UpsamplingNearest2d(scale_factor=2)

        self.relu = nn.ReLU()

    def forward(self, input):
        x, labels = input
        x_0 = x
        x = self.relu(self.chbn_0(x, labels))
        x = self.usample(x)
        x = self.snconv1(x)
        x = self.relu(self.chbn_1(x, labels))
        x = self.snconv2(x)

        x_0 = self.usample(x_0)
        x_0 = self.snconv3(x_0)

        return (x_0 + x, labels)



class Gernerator(nn.Module):
    def __init__(self, z_size, gf_dim, num_classes):
        super(Gernerator, self).__init__()
        self.z_size = z_size
        self.gf_dim = gf_dim
        self.num_classes = num_classes

        self.g_sbh0 = SNLinear(z_size, gf_dim * 16 * 4 * 4)
        self.g_block = nn.Sequential(
            Block(gf_dim * 16, gf_dim * 16, num_classes),   # 8 * 8
            Block(gf_dim * 16, gf_dim * 8, num_classes),    # 16 * 16
            Block(gf_dim * 8, gf_dim * 4, num_classes),     # 32 * 32
            Block(gf_dim * 4, gf_dim * 2, num_classes),     # 64 * 64
            Block(gf_dim * 2, gf_dim, num_classes),         # 128 * 128
        )

        self.g_last = nn.Sequential(
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(),
            SNConv2d(gf_dim, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, zs, target_class):
        act0 = self.g_sbh0(zs)
        act0 = act0.view(-1, self.gf_dim * 16, 4, 4)

        act5, target_class = self.g_block((act0, target_class))
        out = self.g_last(act5)
        return out
