import torch
import torch.nn as nn

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



class ConditionalBatchNorm(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_categories, decay_rate=0.999, center=True, scale=True):
        super(ConditionalBatchNorm, self).__init__(num_categories)
        self.scale = scale
        self.center = center
        self.decay_rate = decay_rate
        self.num_categories = num_categories

    def _check_input_dim(self, input):
        if input.dim() !=4:
            raise ValueError(f'expected 4D input (got {input.dim()}D input)')

    def forward(self, inputs, labels):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

