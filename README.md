# EfficientNet Pytorch Implementation

### Features
- EfficientNet (b0~b7)
- MBConv : Inverted Residual Bottleneck Block
- SE Block
- Swish
- Efficient Unet : Decoder

### Usage
```
from model import EfficientNet, EfficientUnet
model = EfficientNet.from_model('efficientnet-b0')
pretrained_model = EfficientNet.from_pretrained('efficientnet-b0')
unet_model = EfficientUnet.from_model('efficientnet-b0')
unet_pretrained_model = EfficientUnet.from_pretrained('efficientnet-b0')
```
