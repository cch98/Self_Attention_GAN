import torch
import torch.nn as nn

from .components import *




class Block(nn.Module):
    def __init__(self, input_channels, out_channels, name, update_collection=None, downsample=True, act = nn.ReLU):
        super(Block, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.name = name
        self.update_collection = update_collection
        self.downsample = downsample
        self.act = act()

        self.sn_conv1 = SNConv2d(input_channels, out_channels, 3, 1)
        self.sn_conv2 = SNConv2d(out_channels, out_channels, 3, 1)
        if not input_channels == out_channels:
            self.sn_conv3 = SNConv2d(input_channels, out_channels, 1, 1)

        if downsample:
            self.dsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.dsample = None

    def forward(self, x):
        x_0 = x
        x = self.act(x)
        x = self.sn_conv1(x)
        x = self.act(x)
        x = self.sn_conv2(x)

        if self.downsample:
            x = self.dsample(x)

        if self.downsample or not self.input_channels == self.out_channels:
            x_0 = self.sn_conv3(x_0)

        if self.downsample:
            x_0 = self.dsample(x_0)

        return x_0 + x


class OptimizedBlock(nn.Module):
    def __init__(self, input_channels, out_channels, update_collection=None, act=nn.ReLU):
        super(OptimizedBlock, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.update_collection = update_collection
        self.act = act()

        self.sn_conv1 = SNConv2d(input_channels, out_channels, 3, 1)
        self.sn_conv2 = SNConv2d(out_channels, out_channels, 3, 1)
        self.sn_conv3 = SNConv2d(input_channels, out_channels, 1, 1)

        self.dsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_0 = x
        x = self.sn_conv1(x)
        x = self.act(x)
        x = self.sn_conv2(x)
        x = self.dsample(x)

        x_0 = self.dsample(x_0)
        x_0 = self.sn_conv3(x_0)

        return x_0 + x


class Discriminator(nn.Module):
    def __init__(self, input_channels, df_dim, number_classes, update_collection=None, act=nn.ReLU):
        super(Discriminator, self).__init__()
        self.input_channels = input_channels
        self.df_dim = df_dim
        self.number_classes = number_classes
        self.update_collection = update_collection

        self.act = act()

        self.d_block = nn.Sequential(
            OptimizedBlock(input_channels, df_dim, update_collection, act),
            Block(df_dim, df_dim * 2, update_collection, act),
            Block(df_dim * 2, df_dim * 4, update_collection, act),
            Block(df_dim * 4, df_dim * 8, update_collection, act),
            Block(df_dim * 8, df_dim * 16, update_collection, act),
            Block(df_dim * 16, df_dim * 16, update_collection, act),
            act(),
        )

        self.d_sn_linear = SNLinear(df_dim * 16, 1)

        self.embedding_map = torch.Tensor(number_classes, df_dim * 16)

    def forward(self, image, labels):
        x = self.d_block(image)         # B, df_dim*16

        h6 = torch.sum(x, dim=(2, 3))   # B, df_dim*16
        output = self.d_sn_linear(x)    # B, 1

        ###############unclear###########
        # Original code:
        # h_labels = ops.sn_embedding(labels, number_classes, df_dim * 16,
        #                             update_collection=update_collection,
        #                             name='d_embedding')
        #
        onehot = torch.zeros(output.shape[0], self.number_classes)          # B, number_classes
        onehot.scatter(1, labels.view(-1, 1).long(), 1)                     # B, number_classes

        embedding_map_bar = spectral_norm(self.embedding_map, dim=1)        # df_dim*16, number_classes
        h_labels = torch.matmul(onehot, embedding_map_bar)                  # B, df_dim*16
        #################################
        output = output + torch.sum(h6 * h_labels, dim=1, keepdim=True)

        return output


class DiscriminatorTest(nn.Module):
    def __init__(self, input_channels, df_dim, number_classes, update_collection=None, act=nn.ReLU):
        super(DiscriminatorTest, self).__init__()
        self.input_channels = input_channels
        self.df_dim = df_dim
        self.number_classes = number_classes
        self.update_collection = update_collection

        self.act = act()

        self.d_block = nn.Sequential(
            OptimizedBlock(input_channels, df_dim, update_collection, act),
            Block(df_dim, df_dim * 2, update_collection, act),
            SNNonLocalBlockSIM(df_dim * 2),
            Block(df_dim * 2, df_dim * 4, update_collection, act),
            Block(df_dim * 4, df_dim * 8, update_collection, act),
            Block(df_dim * 8, df_dim * 16, update_collection, act),
            Block(df_dim * 16, df_dim * 16, update_collection, act),
            act(),
        )

        self.d_sn_linear = SNLinear(df_dim * 16, 1)

        self.embedding_map = torch.Tensor(number_classes, df_dim * 16)

    def forward(self, image, labels):
        x = self.d_block(image)         # B, df_dim*16

        h6 = torch.sum(x, dim=(2, 3))   # B, df_dim*16
        output = self.d_sn_linear(x)    # B, 1

        ###############unclear###########
        # Original code:
        # h_labels = ops.sn_embedding(labels, number_classes, df_dim * 16,
        #                             update_collection=update_collection,
        #                             name='d_embedding')
        #
        onehot = torch.zeros(output.shape[0], self.number_classes)          # B, number_classes
        onehot.scatter(1, labels.view(-1, 1).long(), 1)                     # B, number_classes

        embedding_map_bar = spectral_norm(self.embedding_map, dim=1)        # df_dim*16, number_classes
        h_labels = torch.matmul(onehot, embedding_map_bar)                  # B, df_dim*16
        #################################
        output = output + torch.sum(h6 * h_labels, dim=1, keepdim=True)

        return output