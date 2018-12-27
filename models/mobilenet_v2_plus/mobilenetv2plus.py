## modified from https://github.com/ansleliu/LightNet

import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mobilenet_v2_plus.bn import InPlaceABN, InPlaceABNWrapper
from models.mobilenet_v2_plus.misc import SCSEBlock, ASPPInPlaceABNBlock, InvertedResidual, conv_bn


class MobileNetV2Plus(nn.Module):

    def load_pretrain(self, pretrain_file):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())

        for key in keys:
            if any(s in key for s in ['.num_batches_tracked', 'logit']):
                continue

        self.load_state_dict(state_dict)


    def __init__(self, n_class=340, in_size=(256, 256), width_mult=1.,
                 out_sec=256, aspp_sec=(12, 24, 36), norm_act=InPlaceABN):
        """
        MobileNetV2Plus: MobileNetV2 based Semantic Segmentation
        :param n_class:    (int)  Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param width_mult: (float) Network width multiplier
        :param out_sec:    (tuple) Number of the output channels of the ASPP Block
        :param aspp_sec:   (tuple) Dilation rates used in ASPP
        """
        super(MobileNetV2Plus, self).__init__()

        self.n_class = n_class
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 1, 1, 1],    # 1/2
            [6, 24, 2, 2, 1],    # 1/4
            [6, 32, 3, 2, 1],    # 1/8
            [6, 64, 4, 2, 2],    # 1/8
            [6, 96, 3, 1, 4],    # 1/8
            [6, 160, 3, 2, 8],   # 1/8
            [6, 320, 1, 1, 16],  # 1/8
        ]

        # building first layer
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0

        self.input_size = in_size

        input_channel = int(32 * width_mult)
        self.mod1 = nn.Sequential(OrderedDict([("conv1", conv_bn(inp=3, oup=input_channel, stride=2))]))

        # building inverted residual blocks
        mod_id = 0
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)

            # Create blocks for module
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(("block%d" % (block_id + 1), InvertedResidual(inp=input_channel,
                                                                                oup=output_channel,
                                                                                stride=s,
                                                                                dilate=1,
                                                                                expand_ratio=t)))
                else:
                    blocks.append(("block%d" % (block_id + 1), InvertedResidual(inp=input_channel,
                                                                                oup=output_channel,
                                                                                stride=1,
                                                                                dilate=d,
                                                                                expand_ratio=t)))

                input_channel = output_channel

            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1

        # building last several layers
        org_last_chns = (self.interverted_residual_setting[0][1] +
                         self.interverted_residual_setting[1][1] +
                         self.interverted_residual_setting[2][1] +
                         self.interverted_residual_setting[3][1] +
                         self.interverted_residual_setting[4][1] +
                         self.interverted_residual_setting[5][1] +
                         self.interverted_residual_setting[6][1])

        self.last_channel = int(org_last_chns * width_mult) if width_mult > 1.0 else org_last_chns
        self.out_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))

        # building classifier
        self.logit = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # channel_shuffle: shuffle channels in groups
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batch_size, groups, channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)

        return x

    def forward(self, x):

        batch_size,C,H,W = x.shape
        mean=[0.485, 0.456, 0.406] #rgb
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x[:,[0]]-mean[0])/std[0],
            (x[:,[1]]-mean[1])/std[1],
            (x[:,[2]]-mean[2])/std[2],
        ],1)

        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoder: feature extraction
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        stg1 = self.mod1(x)     # (N, 32,   224, 448)  1/2
        stg1 = self.mod2(stg1)  # (N, 16,   224, 448)  1/2 -> 1/4 -> 1/8
        stg2 = self.mod3(stg1)  # (N, 24,   112, 224)  1/4 -> 1/8
        stg3 = self.mod4(stg2)  # (N, 32,   56,  112)  1/8
        stg4 = self.mod5(stg3)  # (N, 64,   56,  112)  1/8 dilation=2
        stg5 = self.mod6(stg4)  # (N, 96,   56,  112)  1/8 dilation=4
        stg6 = self.mod7(stg5)  # (N, 160,  56,  112)  1/8 dilation=8
        stg7 = self.mod8(stg6)  # (N, 320,  56,  112)  1/8 dilation=16

        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, ceil_mode=True)    
        stg1_2 = F.max_pool2d(input=stg1_1, kernel_size=3, stride=2, ceil_mode=True)  
        stg1_3 = F.max_pool2d(input=stg1_2, kernel_size=3, stride=2, ceil_mode=True)  
        stg1_4 = F.max_pool2d(input=stg1_3, kernel_size=3, stride=2, ceil_mode=True)  
        
        stg2_1 = F.max_pool2d(input=stg2, kernel_size=3, stride=2, ceil_mode=True)    
        stg2_2 = F.max_pool2d(input=stg2_1, kernel_size=3, stride=2, ceil_mode=True)   
        stg2_3 = F.max_pool2d(input=stg2_2, kernel_size=3, stride=2, ceil_mode=True)    

        stg3_1 = F.max_pool2d(input=stg3, kernel_size=3, stride=2, ceil_mode=True)    
        stg3_2 = F.max_pool2d(input=stg3_1, kernel_size=3, stride=2, ceil_mode=True)    

        stg4_1 = F.max_pool2d(input=stg4, kernel_size=3, stride=2, ceil_mode=True)    

        stg5_1 = F.max_pool2d(input=stg5, kernel_size=3, stride=2, ceil_mode=True)    

        # print('x', x.size())
        # print('stg1', stg1.size())
        # print('stg2', stg2.size())
        # print('stg3', stg3.size())
        # print('stg4', stg4.size())
        # print('stg5', stg5.size())
        # print('stg6', stg6.size())
        # print('stg7', stg7.size())
        

        # print('stg1_4', stg1_4.size())
        # print('stg2_3', stg2_3.size())
        # print('stg3_2', stg3_2.size())
        # print('stg4_1', stg4_1.size())
        # print('stg5_1', stg5_1.size())

        # (N, 712, 56,  112)  1/8  (16+24+32+64+96+160+320)
        stg8 = self.out_se(torch.cat([stg1_4, stg2_3, stg3_2, stg4_1, stg5_1, stg6, stg7], dim=1))
        # stg8 = torch.cat([stg3, stg4, stg5, stg6, stg7, stg1_2, stg2_1], dim=1)

        # print('stg8', stg8.size())

        x = F.adaptive_avg_pool2d(stg8, output_size=1).view(batch_size,-1)

        x = F.dropout(x, p=0.20, training=self.training)
        logit = self.logit(x)

        # print('x', x.size())
        # print('logit', logit.size())

        return logit

    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False


                        


if __name__ == '__main__':
    import time
    import torch
    from torch.autograd import Variable

    # dummy_in = Variable(torch.randn(1, 3, 448, 896).cuda(), requires_grad=True)
    dummy_in = Variable(torch.randn(1, 3, 256, 256).cuda(), requires_grad=True)

    model = MobileNetV2Plus(n_class=10)
    # print(model.state_dict().keys())
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    dummy_out = model(dummy_in)

    model_dict = model.state_dict()

    pre_weight = torch.load("/home/jun/quick-draw/pretrained_models/cityscapes_mobilenetv2_best_model.pkl")["model_state"]

    keys = list(model_dict.keys())

    for key in keys:
        if any(s in key for s in ['logit', '.num_batches_tracked']):
            continue
        print(key)

        model_dict[key] = pre_weight[key]
