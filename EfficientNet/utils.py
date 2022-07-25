import math
import collections
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial

GlobalParams = collections.namedtuple("GlobalParams", 
[
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'batch_norm', 'use_se',
    'se_coefficient', 'local_pooling', 'condconv_num_experts',
    'clip_projection_output', 'blocks_args', 'fix_head_stem', 'use_bfloat16'
])

# 기본값 모두 None
GlobalParams.__new__.__defaults__  = (None, ) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', 
[
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type', 'fused_conv',
    'space2depth', 'condconv', 'activation_fn'
])
BlockArgs.__new__.__defaults__ = (None, ) * len(BlockArgs._fields)

if hasattr(nn, 'SiLU'):
    Swish = nn.SiLU
else:
    class Swish(nn.SiLU):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SwishImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1- sigmoid_i)))
    
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImpl.apply(x)

def round_filters(filters, global_params):
    """ Caculate and round number of filters based on width multiplier.
    Use width_coefficient, depth_divisor and min_depth of global_params.

    Args:
        Filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.
    
    Returns:
        new_filters: New filters number after caculating.
    """

    multiplier = global_params.width_coeffcient
    if not multiplier:
        return filters
    
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor # Pay attention to this line when using min_depth

    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    """ Calculate module's repeat number of a block based on depth multiplier.
        Use depth_coefficient of global_params.
    
    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.
    
    Returns:
        new repeat: New repeat number after calculating.
    """
    
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

def drop_connect(inputs, p, training):
    """Drop the entire conv with givien survival probability.

    Args:
        input (tensor: BCHW): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.
    """

    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs
    
    batch_size = inputs.shape[0]
    survival_prob = 1. - p
    
    random_tensor = survival_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtpe=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    # Unlike conventional way that multiply survival_prob at test time,
    # here we divide survival_prob at training time, such that no addition compute is needed at test time.
    output = inputs * binary_tensor / survival_prob
    return output

def get_width_and_height_from_size(x):
    if isinstance(x, int):
        return x, x
    elif isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.

    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.
    
    Returns:
        output_image_size: A list [h, w]
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]



class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolution, for a dynamic image size.
        The padding is operated in forward function by calculating dynamically.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
    
    def forward(self, x):
        # B C H W
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions
        The Padding module is calculated in construction function, then used in forward.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()
        
    
    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.
    Args:
        image_size (int or tuple): Size of the image.
    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """2D MaxPooling
        The padding is operated in forward function by calculating dynamically.
    """

    def __init__(self, kernel_size, stride, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)

class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    def __init__(self, kernel_size, stride, image_size=None, **kwargs):
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()
    
    def forward(self, x):
        x = self.static_padding(x)
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
        return x
    
def get_same_padding_maxPool2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.
    Args:
        image_size (int or tuple): Size of the image.
    Returns:
        MaxPool2dDynamicSamePadding or MaxPool2dStaticSamePadding.
    """
    if image_size is None:
        return MaxPool2dDynamicSamePadding
    else:
        return partial(MaxPool2dStaticSamePadding, image_size=image_size)