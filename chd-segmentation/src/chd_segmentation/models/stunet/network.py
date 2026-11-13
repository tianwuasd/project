# 来源：uni-medical/STU-Net，Apache-2.0；上游结构 blob 85587aca5fdf1180dc266a138f3ae97c231875b2。
import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self, deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision


class BasicResBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel, padding, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(inputs, outputs, kernel, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(outputs, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(outputs, outputs, kernel, padding=padding)
        self.norm2 = nn.InstanceNorm3d(outputs, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(inputs, outputs, 1, stride=stride) if use_1x1conv else None

    def forward(self, value):
        residual = self.act1(self.norm1(self.conv1(value)))
        residual = self.norm2(self.conv2(residual))
        if self.conv3 is not None:
            value = self.conv3(value)
        return self.act2(residual + value)


class UpsampleLayer(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.conv = nn.Conv3d(inputs, outputs, 1)
        self.pool_op_kernel_size = stride

    def forward(self, value):
        return self.conv(F.interpolate(value, scale_factor=self.pool_op_kernel_size, mode="nearest"))


class STUNet(nn.Module):
    def __init__(self, num_classes, strides, kernels, deep_supervision=True):
        super().__init__()
        dims = [16, 32, 64, 128, 256, 256]
        self.decoder = Decoder(deep_supervision)
        self.conv_blocks_context = nn.ModuleList()
        for index, channels in enumerate(dims):
            block = BasicResBlock(1 if index == 0 else dims[index - 1], channels, kernels[index],
                                  [x // 2 for x in kernels[index]], 1 if index == 0 else strides[index - 1], True)
            self.conv_blocks_context.append(nn.Sequential(block))
        self.upsample_layers = nn.ModuleList()
        self.conv_blocks_localization = nn.ModuleList()
        self.seg_outputs = nn.ModuleList()
        for index in range(5):
            channels = dims[-2 - index]
            self.upsample_layers.append(UpsampleLayer(dims[-1 - index], channels, strides[-1 - index]))
            self.conv_blocks_localization.append(nn.Sequential(BasicResBlock(
                channels * 2, channels, kernels[-2 - index], [x // 2 for x in kernels[-2 - index]], use_1x1conv=True)))
            self.seg_outputs.append(nn.Conv3d(channels, num_classes, 1))

    def forward(self, image):
        skips, outputs = [], []
        for block in self.conv_blocks_context[:-1]:
            image = block(image)
            skips.append(image)
        image = self.conv_blocks_context[-1](image)
        for upsample, block, output, skip in zip(self.upsample_layers, self.conv_blocks_localization,
                                                self.seg_outputs, reversed(skips)):
            image = block(torch.cat((upsample(image), skip), dim=1))
            outputs.append(output(image))
        return tuple(reversed(outputs)) if self.decoder.deep_supervision else outputs[-1]
