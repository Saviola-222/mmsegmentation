import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class DeeplabV3PlusHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self,
                 c1_in_channels,
                 c1_channels,
                 dilations=(1, 6, 12, 18),
                 **kwargs):
        super(DeeplabV3PlusHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                bias=False,  # different
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # self.bottleneck = ConvModule(
        #     (len(dilations) + 1) * self.channels,
        #     self.channels,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

        # self.c1_bottleneck = ConvModule(
        #     c1_in_channels,
        #     c1_channels,
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        self.c1_bottleneck = ConvModule(
            c1_in_channels,
            c1_channels,
            1,
            padding=0,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

        # Final segmentation out
        self.final_seg = nn.Sequential(
            ConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                bias=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                bias=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1, bias=False)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]

        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        aspp_fused = self.bottleneck(aspp_outs)
        c1_output = self.c1_bottleneck(inputs[0])

        aspp_fused = resize(
            aspp_fused,
            size=c1_output.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_out = torch.cat([aspp_fused, c1_output], dim=1)
        final_seg_out = self.dsn_final_seg(seg_out)
        return final_seg_out

    def dsn_final_seg(self, feat):
        feat = self.final_seg(feat)
        """Final feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
