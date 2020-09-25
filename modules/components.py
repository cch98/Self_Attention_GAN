import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils import spectral_norm

class SNConv2d(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(SNConv2d, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(input_channels, out_channels, kernel_size, stride, padding))

    def forward(self, input):
        x = self.conv(input)
        return x


class SNLinear(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(SNLinear, self).__init__()
        self.linear = spectral_norm(nn.Linear(input_channels, out_channels))

    def forward(self, input):
        x = self.linear(input)
        return x



class ConditionalBatchNorm(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False)
    self.embed = nn.Embedding(num_classes, num_features * 2)
    self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

  def forward(self, x, y):
    out = self.bn(x)
    gamma, beta = self.embed(y).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out


class SNNonLocalBlockSIM(nn.Module):
    def __init__(self, input_channels):
        super(SNNonLocalBlockSIM, self).__init__()
        self.input_channels = input_channels
        self.sn_conv_theta = SNConv2d(input_channels, input_channels // 8, 1, 1, 0)
        self.sn_conv_phi = SNConv2d(input_channels, input_channels // 8, 1, 1, 0)
        self.sn_conv_g = SNConv2d(input_channels, input_channels // 2, 1, 1, 0)
        self.sn_conv_attn = SNConv2d(input_channels//2, input_channels, 1, 1, 0)
        self.sigma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        theta = self.sn_conv_theta(x)
        theta = theta.view(B, C//8, -1)

        phi = self.sn_conv_phi(x)
        phi = F.max_pool2d(phi, kernel_size=2, stride=2)
        phi = phi.view(B, C//8, -1)
        phi = phi.transpose(1, 2)

        attn = torch.bmm(phi, theta)
        attn = F.softmax(attn)

        g = self.sn_conv_g(x)
        g = F.max_pool2d(g, kernel_size=2, stride=2)
        g = g.view(B, C//2, -1)
        g = g.transpose(1, 2)

        attn_g = torch.bmm(g, attn)
        attn_g = attn_g.view(B, C//2, H, W)
        attn_g = self.sn_conv_attn(attn_g)

        return x + self.sigma * attn_g