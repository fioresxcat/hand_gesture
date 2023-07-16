import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class Conv2dSiLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        silu = nn.SiLU(inplace=True)

        if use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super(Conv2dSiLU, self).__init__(conv, bn, silu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
    
    
class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)
    

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dSiLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dSiLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)


    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
    

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dSiLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dSiLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetPlusPlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                    )
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
        return dense_x[f"x_{0}_{self.depth}"]
    





class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=3, stride=1, padding=1, act=nn.Identity()):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_c)
        self.act = act
    
    def forward(self, input):
        return self.act(self.bn(self.conv(input)))


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.stride = stride
        self.conv1 = ConvBlock(in_c, out_c, stride=stride, act=nn.ReLU())
        self.conv2 = ConvBlock(out_c, out_c)
        if stride > 1:
            self.downsample = ConvBlock(in_c, out_c, stride=stride)
        
    def forward(self, input):
        x = self.conv1(input)
        x = F.dropout(x, p=0.1)
        x = self.conv2(x)
        if self.stride > 1:
            identity = self.downsample(input)
        else:
            identity = input
        x = x + identity
        return F.relu(x)
    

class DownSampleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res1 = ResBlock(in_c, out_c, stride=2)
        self.dropout = nn.Dropout(p=0.1)
        self.res2 = ResBlock(out_c, out_c)
    
    def forward(self, input):
        return self.res2(self.dropout(self.res1(input)))


class UpSampleBlock(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.res1 = ResBlock(in_c, in_c)
        self.res2 = ResBlock(in_c, in_c)
        self.conv = ConvBlock(in_c, out_c, act=nn.ReLU())

    def forward(self, input):
        x = F.dropout(self.res1(input), p=0.1)
        x = F.dropout(self.res2(x), p=0.1)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        x = self.conv(x)
        return x


class ConnectBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res = ResBlock(in_c=in_c, out_c=out_c)

    def forward(self, input):
        x = self.res(input)
        x = F.dropout(x, p=0.1)
        return x
    




if __name__ == '__main__':
    from pathlib import Path
    import os
    import shutil

    dir = '/data3/users/tungtx2/hand_gesture/classification_data'
    for img_fp in Path(dir).rglob('*.png'):
        if 'masked' in str(img_fp):
            os.remove(img_fp)
            print(f'removed {img_fp}')
    