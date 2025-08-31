# -*- coding:utf-8 -*-

import math
import copy
from functools import partial
from typing import Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import flop_count, parameter_count
from mambapy.mamba import Mamba, MambaConfig
from mamba.mambablock import (CrossMambaFusion_SS2D_SSM, Cross_Mamba_Attention_SSM,
                        CrossConcatMambaFusion, CrossMambaFusionBlock,
                        ConcatMambaFusionBlock, SpectralMamba, Conv1d, VSSBlock, DWConv)

def conv_bn_relu(in_channel, out_channel,kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),  # todo: paddint
        nn.BatchNorm2d(out_channel, momentum=0.9, eps=0.001),  # note 默认可以修改
        nn.ReLU()
    )

def conv_bn_relu_max(in_channel, out_channel,kernel_size=3, stride=1, padding=1,max_kernel=2):
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(max_kernel),
        )

class MultimodalClassier(nn.Module):
    def __init__(self, l1, l2, dim, num_classes):
        super().__init__()
      
        self.conv_lidar= conv_bn_relu(l2, l1, 1, 1, 0)
        self.vssblock1 = VSSBlock(dim=dim, drop_path=0.1, d_state=16, mlp_ratio=2.0)
        self.vssblock2 = VSSBlock(dim=dim, drop_path=0.1, d_state=16, mlp_ratio=2.0)
        self.vssblock3 =  VSSBlock(dim=dim, drop_path=0.1, d_state=16, mlp_ratio=2.0)
        self.vssblock4 =  VSSBlock(dim=dim, drop_path=0.1, d_state=16, mlp_ratio=2.0)

   
        self.cross_mamba0 = CrossMambaFusionBlock(dim=dim, drop_path=0.1, d_state=4, mlp_ratio=2.0)
    
        self.concatmambafusion0 = ConcatMambaFusionBlock(dim=dim, drop_path=0.1, mlp_ratio=2.0, d_state=16)
       
        self.classifier = nn.Sequential(OrderedDict(
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
        ))
        self.linear = nn.Linear(dim, num_classes, bias=False)
       

    def forward(self, x1, x2):  # x1 hsi (64, 144, 11, 11) x2 lidar (64, 1, 11, 11)
        

        x2 = self.conv_lidar(x2) #(64, 144, 11, 11)
      
        x3_init = self.vssblock1(x1)
        x4_init = self.vssblock1(x2)

        x3 = x3_init
        x4 = x4_init

        x5_init = self.vssblock2(x3)
        x6_init = self.vssblock2(x4)
        
        x5 = x5_init
        x6 = x6_init

        x7_init = self.vssblock3(x5)
        x8_init = self.vssblock3(x6)

        x7 = x7_init
        x8 = x8_init

        x9_init = self.vssblock4(x7)
        x10_init = self.vssblock4(x8)

        x11_trans, x12_trans = self.cross_mamba0(x9_init.permute(0, 2, 3, 1).contiguous(), x10_init.permute(0, 2, 3, 1).contiguous())
        x11_trans = x11_trans.permute(0, 3, 1, 2).contiguous()
        x12_trans = x12_trans.permute(0, 3, 1, 2).contiguous()

        x11 = x11_trans
        x12 = x12_trans

        x_add = self.concatmambafusion0(x11.permute(0, 2, 3, 1).contiguous(), x12.permute(0, 2, 3, 1).contiguous())
        x_add = x_add.permute(0, 3, 1, 2).contiguous()

        out = self.classifier(x_add)

     
        out = self.linear(out)
        return out

