from email.encoders import encode_noop
from turtle import forward
from typing import OrderedDict
from jwt import DecodeError
from numpy import block
from pytest import skip
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
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1. - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0. < self._block_args.se_ratio <= 1.)
        self.id_skip = block_args.id_skip

        #Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio # number of output channels
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
            num_sqeezed_channels = max(1, int(oup * self._block_args.se_ratio))
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
            if drop_connect_rate > 0.:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x
    

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


        
class EfficientNet(nn.Module):
    """EfficientNet Model.
        Most easily loaded with the .from_name or .from_retraiend methods.

    Args:
        blocks_args (list[namedtupe]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    
    Examples:
        from efficientnet.model import Efficientnet
        model = EfficientNet.from_pretrained('efficientnet-b0)
        model.eval()
    """
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch Norm parameters
        bn_mon = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = self._global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3 # rgb
        out_channels = round_filters(32, self._global_params) # number of output channels
        self._conv_stem = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1 : #modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))

        # Head
        in_channels = block_args.output_filters # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mon, eps=bn_eps)

        #Final linear layer
        if self._global_params.include_top:
            self._avg_pooling = nn.AdaptiveAvgPool2d(1)
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        
        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()
    
    def set_swish(self, memory_efficient=True):
        """Set Swish function as memory as efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use Convolution layer to extract features
            from reduction levels i in [1, 2, 3, 4, 5].
        
        Args:
            inputs (tensor): Input tensor.
        
        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
        """
        endpoints = dict()

        # stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        #Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x
            
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        return endpoints

    def extract_features(self, inputs):
        """Use convolution layer to extract feature.

        Args:
            inputs (tensor): Input tensor
        
        Returns:
            Output of the final convolution
            layer in the efficient model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function
            Calls extract_features to extract features, applies final linear layer, and returns logits.

            Args:
                inputs (tensor): Input tensor.
            
            Returns:
                Output of this model after processing.
        """

        # Convolution layers
        x = self.extract_features(inputs)
        
        #Pooling and final linear layer
        if self._global_params.include_top:
            x = self._avg_pooling(x)
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    
    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for EfficientNet
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An EfficientNet Model.
        """
    
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False, in_channels=3, num_classes=1000, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights trained with advprop (valid when weights_path is None).
            in_channels (int):
                Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """

        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @property
    def image_size(self):
        return self._global_params.image_size
    
    @property
    def name(self):
        return self._global_params.model_name
    
    @property
    def each_channels(self):
        round_filters()


    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.
        
        Returns:
            bool: Is a valid name or not
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of : ' + ', '.join(VALID_MODELS))
    

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """

        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, biase=False)

class EfficientUnet(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args=blocks_args, global_params=global_params)
        
        skip_channels = self.extract_skip_channels()
        last_channels = round_filters(1280, self._global_params)
        bn_mom = 1. - global_params.batch_norm_momentum
        bn_eps = global_params.batch_norm_epsilon
        prev_skip_channel = skip_channels[-1]
        self._decoder_blocks = nn.ModuleList([
            DecodeBlock(in_channels=last_channels, skip_channels=prev_skip_channel, out_channels=prev_skip_channel, bn_mom=bn_mom, bn_eps=bn_eps)
        ])

        for skip_channel in reversed(skip_channels[:-1]):
            self._decoder_blocks.append(
                DecodeBlock(in_channels=prev_skip_channel, skip_channels=skip_channel, out_channels=skip_channel, bn_mom=bn_mom, bn_eps=bn_eps)
            )
            prev_skip_channel = skip_channel
        
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        self._conv_top = Conv2d(in_channels=skip_channel, out_channels=self._global_params.num_classes, kernel_size=1)
        
    def extract_features(self, inputs):
        skip_conn = []

        x = inputs
        skip_conn.append(x)

        x = self._swish(self._bn0(self._conv_stem(x)))
        prev_x = x
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2): # if reduce
                skip_conn.append(prev_x)
            prev_x = x
        
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x, skip_conn

    def extract_skip_channels(self):
        image_size = self._global_params.image_size
        out_channels = [3,]

        # stem
        image_size = calculate_output_image_size(image_size, 2)

        for block_args in self._blocks_args:
            new_image_size = calculate_output_image_size(image_size, block_args.stride)
            if image_size != new_image_size:
                out_channels.append(round_filters(block_args.input_filters, self._global_params))
        return out_channels

    def forward(self, inputs):
        x = inputs
        x, skip_conns = self.extract_features(x)
        for block, skip_conn in zip(self._decoder_blocks, reversed(skip_conns)):
            x = block(x, skip_conn)
        
        x = self._conv_top(x)
        return x
        
class DecodeBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bn_mom, bn_eps, image_size=None):
        super().__init__()
        Conv2d = get_same_padding_conv2d(image_size)
        self._conv0 = Conv2d(in_channels=in_channels+skip_channels, out_channels=out_channels, kernel_size=3)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._swish = MemoryEfficientSwish()
    
    def forward(self, inputs, skip=None):
        x = inputs
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.concat([x, skip], dim=1)
        x = self._swish(self._bn0(self._conv0(x)))
        x = self._swish(self._bn1(self._conv1(x)))
        return x
