import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F

import numpy as np
from models.BEM import ULite


class DCONet(nn.Module):
    def __init__(self, stage_num=6, slayers=6, llayers=3, mlayers=2, channel=32, mode='train'):
        super(DCONet, self).__init__()
        self.stage_num = stage_num
        self.decos = nn.ModuleList()
        self.mode = mode
        for _ in range(stage_num):
            self.decos.append(DecompositionModule(slayers=slayers, llayers=llayers,
                                                  mlayers=mlayers, channel=channel))
        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, D):
        T = torch.zeros(D.shape).to(D.device)
        #初始化h和c
        h, c = None, None
        #在每一轮的传递中都有h和c
        for i in range(self.stage_num):
            D, T,h,c = self.decos[i](D, T,h,c)
        if self.mode == 'train':
            return D,T
        else:
            return T

class DecompositionModule(object):
    pass


class DecompositionModule(nn.Module):
    def __init__(self, slayers=6, llayers=3, mlayers=2, channel=32):
        super(DecompositionModule, self).__init__()
        self.lowrank = LowrankModule(channel=channel, layers=llayers)
        self.sparse = SparseModule(channel=channel, layers=slayers)
        self.merge = MergeModule(channel=channel, layers=mlayers)

    def forward(self, D, T,h,c):
        B = self.lowrank(D, T)
        T,h,c = self.sparse(D, B, T,h,c)
        D = self.merge(B, T)
        return D, T,h,c


class LowrankModule(nn.Module):
    def __init__(self, channel=32, layers=3):
        super(LowrankModule, self).__init__()

        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        #self.relu = nn.ReLU()
        #self.gamma = nn.Parameter(torch.Tensor([0.02]), requires_grad=True)
        self.unet=ULite()

    def forward(self, D, T):
        x = D - T
        # B = x + self.convs(x)
        B = self.unet(x)
        return B


class SparseModule(nn.Module):
    def __init__(self, channel=32, layers=6):
        super(SparseModule, self).__init__()
        #self.att = ESSAttn(1)
        # 第一层stage之后的状态变量的卷积层
        self.conv_x = nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1)  # 处理输入x（D - B）
        self.conv_h = nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1)  # 处理隐藏状态h（T）
        self.conv_c = nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1)
        #第二层stage之后的状态变量的卷积层
        self.conv_x2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1)  # 处理输入x（D - B）
        self.conv_h2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1)  # 处理隐藏状态h（T）
        self.conv_c2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1)
        # 遗忘门
        self.conv_f = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1)

        # 输入门
        self.conv_i = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1)

        # 输出门
        self.conv_o = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1)

        # 细胞状态更新的卷积层
        self.cell_state = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1)

        # 输出层
        self.conv_out = nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1)
        self.epsilon_per_channel = nn.Parameter(torch.ones(channel) * 0.01, requires_grad=True)
        # 定义epsilon为一个参数，用于稀疏更新的控制
        #self.epsilon = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

        # 初始化卷积层权重
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, D, B, T,h,c):
        x = D - B +T
        # 确保h和c不为None，若为None，则初始化为零
        if h is None:
            h = torch.zeros_like(D)  # 依据D的形状初始化
            c = torch.zeros_like(D)
            h = self.conv_h(h)
            c = self.conv_c(c)
            x1 = self.conv_x(x)
        # if c is None:
        #     c = torch.zeros_like(D)  # 依据D的形状初始化
        # x 作为当前的输入 D - B
        else:
            x1 = self.conv_x(x)
            # h 作为上一时刻的隐藏状态 T
            h = self.conv_h(h)
            c = self.conv_c2(c)
        # 循环进行多轮更新
        #cell = T
        for _ in range(1):  # 这里设置循环进行2次更新
            # 计算遗忘门、输入门、输出门
            combine= x1
            f = torch.sigmoid(self.conv_f(combine))
            i = torch.sigmoid(self.conv_i(combine))
            o = torch.sigmoid(self.conv_o(combine))

            # 细胞状态的更新
            c = f * c + i * torch.tanh(self.cell_state(x1))

            # 计算新的隐藏状态
            h = o * torch.tanh(c)
            h = h * self.epsilon_per_channel.view(1, -1, 1, 1)
            #h=self.epsilon * self.conv_out(h)
            h=self.conv_out(h)
            # 最终的稀疏更新
            T_new = x - h
            # 更新隐藏状态
            #h = T_new  # 使用 T_new 作为下一次更新的 h

        return T_new,h,c


class MergeModule(nn.Module):

    def __init__(self, channel=32, layers=2):
        super(MergeModule, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                  nn.BatchNorm2d(channel),
                  nn.ReLU(True)]
        for i in range(layers):
             convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
             convs.append(nn.BatchNorm2d(channel))
             convs.append(nn.ReLU(True))
        # convs.append(ScConv(channel))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.mapping = nn.Sequential(*convs)
        #self.unet1=ULite()


    def forward(self, B, T):
        x = B + T
        #D=self.unet1(x)
        D = self.mapping(x)

        return D


def main():
    # 设置输入张量的形状：假设输入大小为 (batch_size=1, channels=1, height=64, width=64)
    D = torch.randn(1, 3, 64, 64)  # 输入张量 (1, 1, 64, 64)

    # 创建RPCANet实例
    model = RPCANet(stage_num=6, slayers=6, llayers=3, mlayers=3, channel=32, mode='train')

    # 前向传播，获取输出
    D_out, T_out = model(D)

    # 打印每一层的输出形状
    print(f"Input shape: {D.shape}")
    print(f"Output shape (D): {D_out.shape}")
    print(f"Output shape (T): {T_out.shape}")


if __name__ == "__main__":
    main()
