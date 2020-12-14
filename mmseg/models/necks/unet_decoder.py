import torch.nn as nn

from ..backbones.unet import BasicConvBlock, UpConvBlock
from ..builder import NECKS


@NECKS.register_module()
class UNetDecoder(nn.Module):
    """UNet style decoder.
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    https://arxiv.org/pdf/1505.04597.pdf


    Args:
        in_channels (Sequence[int]): Channels of each input feature map.
            Default: (64, 128, 256, 512, 1024).
        dec_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondance decoder stage.
            Default: (2, 2, 2, 2).
        upsamples (Sequence[bool]): Whether need upsample the high-level
            feature map or not. If the size of high-level feature map is the
            same as that of skip feature map (low-level feature map from
            encoder), it does not need upsample the high-level feature map and
            the upsample is False. Default: (True, True, True, True).
        dec_dilations (Sequence[int]): Dilation rate of each stage in decoder.
            Default: (1, 1, 1, 1).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
        dcn (bool): Use deformable convoluton in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.

    """

    def __init__(self,
                 in_channels=(64, 128, 256, 512, 1024),
                 dec_num_convs=(2, 2, 2, 2),
                 upsamples=(True, True, True, True),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 dcn=None,
                 plugins=None):
        super(UNetDecoder, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.decoder = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.decoder.append(
                UpConvBlock(
                    conv_block=BasicConvBlock,
                    in_channels=in_channels[i + 1],
                    skip_channels=in_channels[i],
                    out_channels=in_channels[i],
                    num_convs=dec_num_convs[i],
                    stride=1,
                    dilation=dec_dilations[i],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    upsample=upsamples[i],
                    upsample_cfg=upsample_cfg,
                    dcn=None,
                    plugins=None))

    def forward(self, enc_outs):
        assert len(enc_outs) == len(self.decoder) + 1
        x = enc_outs[-1]
        dec_outs = [x]
        # for i in range(len(self.decoder) - 1, -1, -1):
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        return dec_outs
