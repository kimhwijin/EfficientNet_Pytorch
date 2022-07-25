from doctest import OutputChecker
from random import triangular
from tkinter import image_names
import torch
from torch import nn
from torch.nn import functional as F
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficient_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)


VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 
    'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 
    'efficientnet-b6', 'efficientnet-b7', 'efficientnet-b8'
)

class MBConvBlock(nn.Module):
    """Module Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py
        global_params (namedtuple): GlobalParams, defined in utils.py
        image_size (tuple or list): [image_height, image_width].
    """

    def __init__(self, block_args, global_params, image_size=None):
        self._block_args = block_args
        self._bn_mom = 1. - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0. < self._block_args.se_ratio <= 1.)
        self.id_skip = block_args.id_skip

        #Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratios # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        
        #deep wise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=k, stride=s, bias=False) # groups make it depthwise
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1,1))
            num_sqeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_sqeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_sqeezed_channels, out_channels=oup, kernel_size=1)
        
        # Point wise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()
    
    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward functions.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, 0.0~1.0).

        
        Returns:
            Output of this block after processing.
        """

        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)
        
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
    

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1) # H, W -> 1, 1
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x # not swish
        

        #Pointwise Conv
        x = self._project_conv(x)
        x = self._bn2(x)

        #Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x
    

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


        

