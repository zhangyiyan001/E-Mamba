# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
import time
from engine.logger import get_logger
from mamba.vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock
from collections import OrderedDict
logger = get_logger()


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class RGBXTransformer(nn.Module):
    def __init__(self,
                 in_chans=144,
                 num_classes=15,
                 norm_layer=nn.LayerNorm,
                 depths=[2, 2, 9],  # [2,2,27,2] for vmamba small
                 dims=48,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v2',
                 patch_size=1,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()



        self.vssm = Backbone_VSSM(
            patch_size=patch_size,
            in_chans=in_chans,
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
            out_indices=(0, 1, 2),
        )

        self.cross_mamba = nn.ModuleList(
            CrossMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(3)
        )
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(3)
        )

        # absolute position embedding
        self.classifier = nn.Sequential(OrderedDict(
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(256, num_classes),
        ))
        self.adaptive_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 64 * 2, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(64 * 2),
            Permute(0, 3, 1, 2)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 128 * 2, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(128 * 2),
            Permute(0, 3, 1, 2),
        )

        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 128 * 2, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(128 * 2),
            Permute(0, 3, 1, 2),
        )
        self.convlidar = nn.Sequential(
            nn.Conv2d(1, in_chans, 1, 1, 0),
            nn.BatchNorm2d(in_chans)) #扩充LiDAR通道数


    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x C x H x W
        """
        B = x_rgb.shape[0]
        x_e = self.convlidar(x_e)
        outs_fused = []

        outs_rgb = self.vssm(x_rgb)  # B x C x H x W
        outs_x = self.vssm(x_e)  # B x C x H x W

        for i in range(3):
            out_rgb = outs_rgb[i]
            out_x = outs_x[i]
            # cross attention
            cma = True
            cam = True
            if cma and cam:
                cross_rgb, cross_x = self.cross_mamba[i](out_rgb.permute(0, 2, 3, 1).contiguous(),
                                        out_x.permute(0, 2, 3, 1).contiguous())  # B x H x W x C
                x_fuse = self.channel_attn_mamba[i](cross_rgb, cross_x).permute(0, 3, 1, 2).contiguous()
            elif cam and not cma:
                x_fuse = self.channel_attn_mamba[i](out_rgb.permute(0, 2, 3, 1).contiguous(),
                                        out_x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            elif not cam and not cma:
                x_fuse = (out_rgb + out_x)
            outs_fused.append(x_fuse)
        return outs_fused

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        feature_level0, feature_level1, feature_level2 = out[0], out[1], out[2]
        # feature_level0_1 = self.downsample1(feature_level0)
        # feature_level0_2 = self.downsample2(feature_level0_1)
        # feature_level1 = feature_level1 + feature_level0_1
        # feature_level1_1 = self.downsample3(feature_level1)
        # feature_level2 = feature_level0_2 + feature_level1_1 + feature_level2
        out = self.classifier(feature_level2)

        return out

class vssm_tiny(RGBXTransformer):
    def __init__(self):
        super(vssm_tiny, self).__init__(
            in_chans=144,
            num_classes=15,
            depths=[3, 3, 5],
            dims=64,
            pretrained=None,
            mlp_ratio=2.0,
            downsample_version='v2',
            drop_path_rate=0.1
        )
#
# model = vssm_tiny().cuda()
# x1 = torch.randn(1, 144, 15, 15).cuda()
# x2 = torch.randn(1, 144, 15, 15).cuda()
# out = model(x1, x2)
# print(out.shape)

