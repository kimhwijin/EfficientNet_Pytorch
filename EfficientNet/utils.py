import math
import collections
from numpy import isin
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
import re

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
    

class BlockDecoder(object):
    """Block Decoder for readability.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.
        
        Args:
            block_string (str): A string notation of arguments.
                Examles: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.
        
        Returns:
            BlockArgs: The namedTuple defined at the top of this file.
        """

        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
        
        assert (
            ('s' in options and len(options['s']) == 1) or 
            (len(options['s']) == 2 and options['s'][0] == options['s'][1])
        )

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio = float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string)
        )
    
    @staticmethod
    def _encode_block_string(block):
        """Encode a block to string
        
        Args:
            block (namedTuple): A BlockArgs type argument.
        
        Returns:
            block_strings: A String form of BlockArgs.
        """

        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
        ]

        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        
        return '_'.join(args)
    
    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.

        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.
        
        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """

        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args
    
    @staticmethod
    def encode(blocks_args):
        """Encode a list of BlockArgs to a list of strings.
        
        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs manedtuples of block args.

        Returns:
            block_strings: A list of strings, each string is a notation of block.

        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings
    

