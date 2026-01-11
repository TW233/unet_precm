import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image


class Fconv_PCA(nn.Module):

    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0, padding_mode="zeros"):

        super(Fconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())
        self.ifbias = bias
        if ifIni:
            expand = 1
        else:
            expand = tranNum
        # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
        self.expand = expand
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, Basis.size(3)), requires_grad=True)
        # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        if bias:
            self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1))
        else:
            self.register_parameter('c', None)

        self.padding_mode = padding_mode

        self.reset_parameters()

    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0

            Num = tranNum // expand
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                         for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)

            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])
            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)
        else:
            _filter = self.filter
            if self.ifbias:
                _bias = self.bias

        if self.padding_mode == 'zeros':
            # 如果是默认的 zeros，直接传给 conv2d 即可
            padded_input = input
            conv_padding = self.padding
        else:
            # 如果是 reflect 或 replicate，先使用 F.pad 手动填充
            # F.pad 的参数顺序是 (left, right, top, bottom)
            if isinstance(self.padding, int):
                pad_arg = (self.padding, self.padding, self.padding, self.padding)
            elif isinstance(self.padding, tuple):
                # 假设输入 tuple 是 (pad_h, pad_w)，F.pad 需要 (pad_w, pad_w, pad_h, pad_h)
                pad_arg = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            else:
                raise ValueError("Padding must be int or tuple")

            # 手动执行填充
            padded_input = F.pad(input, pad_arg, mode=self.padding_mode)
            # 既然已经填充过了，卷积时的 padding 设为 0
            conv_padding = 0

        output = F.conv2d(padded_input,
                          _filter,
                          padding=conv_padding,
                          dilation=1,
                          groups=1)
        if self.ifbias:
            output = output + _bias
        return output

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias:
                    del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0
            Num = tranNum // expand
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                         for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)
            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])
            self.register_buffer("filter", _filter)
            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)

        return super(Fconv_PCA, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


class Fconv_PCA_out(nn.Module):

    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):

        super(Fconv_PCA_out, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())

        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, 1, Basis.size(3)), requires_grad=True)
        # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')

        # iniw = Getini_reg(Basis.size(3), inNum, outNum, 1, weight)*iniScale
        # self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        self.ifbias = bias
        if bias:
            self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()

    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijok,mnak->manoij', self.Basis, self.weights)
            _filter = tempW.reshape([outNum, inNum * tranNum, self.sizeP, self.sizeP])
        else:
            _filter = self.filter
        _bias = self.c
        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + _bias

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijok,mnak->manoij', self.Basis, self.weights)

            _filter = tempW.reshape([outNum, inNum * tranNum, self.sizeP, self.sizeP])
            self.register_buffer("filter", _filter)
        return super(Fconv_PCA_out, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)


class Fconv_1X1(nn.Module):

    def __init__(self, inNum, outNum, tranNum=4, ifIni=0, bias=True, padding=0, Smooth=True, iniScale=1.0,
                 padding_mode="zeros"):

        super(Fconv_1X1, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum

        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_reg(1, inNum, outNum, self.expand) * iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)

        self.padding = padding
        self.padding_mode = padding_mode

        # 【修正 1】保存 bias 标志位
        self.ifbias = bias

        # 【修正 2】如果 bias 为 False，将参数注册为 None，而不是创建 CPU 张量
        if bias:
            self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=True)
        else:
            self.register_parameter('c', None)

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = self.weights.unsqueeze(4).unsqueeze(1).repeat([1, tranNum, 1, 1, 1, 1])

        Num = tranNum // expand
        tempWList = [
            torch.cat([tempW[:, i * Num:(i + 1) * Num, :, -i:, ...], tempW[:, i * Num:(i + 1) * Num, :, :-i, ...]],
                      dim=3) for i in range(expand)]
        tempW = torch.cat(tempWList, dim=1)

        _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, 1, 1])

        # 【修正 3】移除这里无条件的 bias 计算
        # bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])

        if self.padding_mode == 'zeros':
            # 如果是默认的 zeros，直接传给 conv2d 即可
            padded_input = input
            conv_padding = self.padding
        else:
            # 如果是 reflect 或 replicate，先使用 F.pad 手动填充
            if isinstance(self.padding, int):
                pad_arg = (self.padding, self.padding, self.padding, self.padding)
            elif isinstance(self.padding, tuple):
                pad_arg = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            else:
                raise ValueError("Padding must be int or tuple")

            padded_input = F.pad(input, pad_arg, mode=self.padding_mode)
            conv_padding = 0

        output = F.conv2d(padded_input,
                          _filter,
                          padding=conv_padding,
                          dilation=1,
                          groups=1)

        # 【修正 4】只有在启用 bias 时才计算并相加
        if self.ifbias:
            # 此时 self.c 是 nn.Parameter，会自动随模型移动到 GPU
            bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
            output = output + bias

        return output


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size, tranNum=4, inP=None,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, Smooth=True, iniScale=1.0):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(
                conv(kernel_size, n_feats, n_feats, tranNum=tranNum, inP=inP, padding=(kernel_size - 1) // 2, bias=bias,
                     Smooth=Smooth, iniScale=iniScale))
            if bn:
                m.append(F_BN(n_feats, tranNum))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return F.relu(res)


def Getini_reg(nNum, inNum, outNum, expand, weight=1):
    A = (np.random.rand(outNum, inNum, expand, nNum) - 0.5) * 2 * 2.4495 / np.sqrt((inNum) * nNum) * np.expand_dims(
        np.expand_dims(np.expand_dims(weight, axis=0), axis=0), axis=0)

    return torch.FloatTensor(A)
    # Float64 时用这个替换 torch.FloatTensor(A)
    # return torch.tensor(A, dtype=torch.get_default_dtype())


def GetBasis_PCA(sizeP, tranNum=8, inP=None, Smooth=True):
    if inP == None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP, tranNum)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    Mask = np.expand_dims(Mask, 2)
    theta = np.arange(tranNum) / tranNum * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)
    #    theta = torch.FloatTensor(theta)
    X = np.cos(theta) * X0 - np.sin(theta) * Y0
    Y = np.cos(theta) * Y0 + np.sin(theta) * X0
    #    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X, 3), 4)
    Y = np.expand_dims(np.expand_dims(Y, 3), 4)
    v = np.pi / inP * (inP - 1)
    p = inP / 2

    k = np.reshape(np.arange(inP), [1, 1, 1, inP, 1])
    l = np.reshape(np.arange(inP), [1, 1, 1, 1, inP])

    BasisC = np.cos((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
    BasisS = np.sin((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)

    BasisC = np.reshape(BasisC, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)
    BasisS = np.reshape(BasisS, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)

    BasisC = np.reshape(BasisC, [sizeP * sizeP * tranNum, inP * inP])
    BasisS = np.reshape(BasisS, [sizeP * sizeP * tranNum, inP * inP])

    BasisR = np.concatenate((BasisC, BasisS), axis=1)

    U, S, VT = np.linalg.svd(np.matmul(BasisR.T, BasisR))

    Rank = np.sum(S > 0.0001)
    BasisR = np.matmul(np.matmul(BasisR, U[:, :Rank]), np.diag(1 / np.sqrt(S[:Rank] + 0.0000000001)))
    BasisR = np.reshape(BasisR, [sizeP, sizeP, tranNum, Rank])

    temp = np.reshape(BasisR, [sizeP * sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis=0) ** 2, axis=0) + np.std(np.sum(temp ** 2 * sizeP * sizeP, axis=0),
                                                              axis=0)) / np.mean(
        np.sum(temp, axis=0) ** 2 + np.sum(temp ** 2 * sizeP * sizeP, axis=0), axis=0)
    Trod = 1
    Ind = var < Trod
    Rank = np.sum(Ind)
    Weight = 1 / np.maximum(var, 0.04) / 25
    if Smooth:
        BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight, 0), 0), 0) * BasisR

    return torch.FloatTensor(BasisR), Rank, Weight
    # Float64 时将 return torch.FloatTensor(BasisR), Rank, Weight 修改为如下：
    # 这样可以保证 BasisR 的类型与 self.weights (由 set_default_dtype 决定) 保持一致
    # return torch.tensor(BasisR, dtype=torch.get_default_dtype()), Rank, Weight


def MaskC(SizeP, tranNum):
    p = (SizeP - 1) / 2
    x = np.arange(-p, p + 1) / p
    X, Y = np.meshgrid(x, x)
    C = X ** 2 + Y ** 2
    if tranNum == 4:
        Mask = np.ones([SizeP, SizeP])
    else:
        if SizeP > 4:
            Mask = np.exp(-np.maximum(C - 1, 0) / 0.2)
        else:
            Mask = np.exp(-np.maximum(C - 1, 0) / 2)
    return X, Y, Mask


class PointwiseAvgPoolAntialiased(nn.Module):

    def __init__(self, sizeF, stride, padding=None):
        super(PointwiseAvgPoolAntialiased, self).__init__()
        sigma = (sizeF - 1) / 2 / 3
        self.kernel_size = (sizeF, sizeF)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        if padding is None:
            padding = int((sizeF - 1) // 2)

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Build the Gaussian smoothing filter
        grid_x = torch.arange(sizeF).repeat(sizeF).view(sizeF, sizeF)
        grid_y = grid_x.t()
        grid = torch.stack([grid_x, grid_y], dim=-1)
        mean = (sizeF - 1) / 2.
        variance = sigma ** 2.
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())
        _filter = torch.exp(r / (2 * variance))
        _filter /= torch.sum(_filter)
        _filter = _filter.view(1, 1, sizeF, sizeF)
        self.filter = nn.Parameter(_filter, requires_grad=False)
        # self.register_buffer("filter", _filter)

    def forward(self, input):
        _filter = self.filter.repeat((input.shape[1], 1, 1, 1))
        output = F.conv2d(input, _filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return output


class F_BN(nn.Module):
    def __init__(self, channels, tranNum=4):
        super(F_BN, self).__init__()
        self.BN = nn.BatchNorm2d(channels)
        self.tranNum = tranNum

    def forward(self, X):
        X = self.BN(X.reshape([X.size(0), int(X.size(1) / self.tranNum), self.tranNum * X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum * X.size(1), int(X.size(2) / self.tranNum), X.size(3)])


class F_Dropout(nn.Module):
    def __init__(self, zero_prob=0.5, tranNum=4):
        # nn.Dropout2d
        self.tranNum = tranNum
        super(F_Dropout, self).__init__()
        self.Dropout = nn.Dropout2d(zero_prob)

    def forward(self, X):
        X = self.Dropout(X.reshape([X.size(0), int(X.size(1) / self.tranNum), self.tranNum * X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum * X.size(1), int(X.size(2) / self.tranNum), X.size(3)])


def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s - 1) / 2
    t = (c - margin / 100. * c) ** 2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r) / sig ** 2)
            else:
                mask[..., x, y] = 1.
    return mask


class MaskModule(nn.Module):

    def __init__(self, S: int, margin: float = 0.):
        super(MaskModule, self).__init__()

        self.margin = margin
        self.mask = torch.nn.Parameter(build_mask(S, margin=margin), requires_grad=False)

    def forward(self, input):
        assert input.shape[2:] == self.mask.shape[2:]

        out = input * self.mask
        return out


class GroupPooling(nn.Module):
    def __init__(self, tranNum):
        super(GroupPooling, self).__init__()
        self.tranNum = tranNum

    def forward(self, input):
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)])
        output = torch.max(output, 2).values
        return output


class GroupMeanPooling(nn.Module):
    def __init__(self, tranNum):
        super(GroupMeanPooling, self).__init__()
        self.tranNum = tranNum

    def forward(self, input):
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)])
        output = torch.mean(output, 2)
        return output


def Getini(sizeP, inNum, outNum, expand):
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    X0 = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(X0, 0), 0), 4), 0)
    y = Y0[:, 1]
    y = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(y, 0), 0), 3), 0)

    orlW = np.zeros([outNum, inNum, expand, sizeP, sizeP, 1, 1])
    for i in range(outNum):
        for j in range(inNum):
            for k in range(expand):
                temp = np.array(
                    Image.fromarray(((np.random.randn(3, 3)) * 2.4495 / np.sqrt((inNum) * sizeP * sizeP))).resize(
                        (sizeP, sizeP)))
                orlW[i, j, k, :, :, 0, 0] = temp

    v = np.pi / sizeP * (sizeP - 1)
    k = np.reshape((np.arange(sizeP)), [1, 1, 1, 1, 1, sizeP, 1])
    l = np.reshape((np.arange(sizeP)), [1, 1, 1, 1, 1, sizeP])

    tempA = np.sum(np.cos(k * v * X0) * orlW, 4) / sizeP
    tempB = -np.sum(np.sin(k * v * X0) * orlW, 4) / sizeP
    A = np.sum(np.cos(l * v * y) * tempA + np.sin(l * v * y) * tempB, 3) / sizeP
    B = np.sum(np.cos(l * v * y) * tempB - np.sin(l * v * y) * tempA, 3) / sizeP
    A = np.reshape(A, [outNum, inNum, expand, sizeP * sizeP])
    B = np.reshape(B, [outNum, inNum, expand, sizeP * sizeP])
    iniW = np.concatenate((A, B), axis=3)

    return torch.FloatTensor(iniW)
    # Float64 时替换 torch.FloatTensor(iniW)
    # return torch.tensor(iniW, dtype=torch.get_default_dtype())



class Fconv_Down(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None, ifIni=0, bias=True, Smooth=True, stride=2):
        super().__init__()

        if inP is None:
            inP = sizeP
        self.sizeP = sizeP
        self.inNum = inNum
        self.outNum = outNum
        self.tranNum = tranNum
        self.ifbias = bias
        if bias:
            self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1))
        else:
            self.register_parameter('c', None)
        if padding is None:
            self.padding = 0
        else:
            self.padding = padding
        if ifIni:
            expand = 1
        else:
            expand =  tranNum
        self.expand = expand

        Basis, Rank, Weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth) # Basis: 四维张量[sizeP, sizeP, tranNum, Rank], Weight: 一维向量[Rank]

        self.register_buffer('Basis', Basis) # [sizeP, sizeP, tranNum, Rank]
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, Rank), requires_grad=True) # [outNum, inNum, expand, Rank]

        self.stride = stride

        self.reset_parameters() # 初始化参数


    def reset_parameters(self) -> None: # 这也是Pytorch官方对Linear层的默认初始化方法，即Kaiming初始化
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.c, -bound, bound)


    # input: [Batch, C_in, H, W] = [Batch, inNum * expand, H, W]
    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            # tempW = torch.einsum('ijok,mnak->monaij', [self.Basis, self.weights])   # for torch<1.0

            Num = tranNum // expand
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                         for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)

            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])
            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)
        else:
            _filter = self.filter
            if self.ifbias:
                _bias = self.bias

        # padding fod different stride
        p = self.padding[0] if isinstance(self.padding, (tuple, list)) else self.padding
        s = self.stride[0] if isinstance(self.stride, (tuple, list)) else self.stride

        input = F.pad(input, (p, p, p, p), mode="constant", value=0)
        if s >= 2:
            H_out, W_out = (input.shape[2] - self.sizeP) // self.stride + 1, (input.shape[3] - self.sizeP) // self.stride + 1
            delta_H, delta_W = input.shape[2] - ((H_out - 1) * self.stride + self.sizeP), input.shape[3] - ((W_out - 1) * self.stride + self.sizeP)

            input[:, 1::4, :, :] = torch.roll(input[:, 1::4, :, :], shifts=-delta_W, dims=-2)
            input[:, 2::4, :, :] = torch.roll(input[:, 2::4, :, :], shifts=-delta_W, dims=-2)
            input[:, 2::4, :, :] = torch.roll(input[:, 2::4, :, :], shifts=-delta_H, dims=-1)
            input[:, 3::4, :, :] = torch.roll(input[:, 3::4, :, :], shifts=-delta_H, dims=-1)

        output = F.conv2d(input, _filter,
                          stride=s,
                          padding=0,
                          dilation=1,
                          groups=1)
        if self.ifbias:
            output = output + _bias
        return output

    # 重写nn.Module的train方法, 实现缓存的优化
    def train(self, mode=True):
        # 在训练mode时, 由于每次反向传播之后self.weights都会更新, 因此每次forward调用就要根据最新的weights来重新计算新的卷积核_filter
        # 如果存在之前缓存的filter和bias, 就删掉, 强制forward函数进行重新计算
        if mode:
            if hasattr(self, 'filter'):
                del self.filter
                if self.ifbias:
                    del self.bias

        # mode=False && self.training=True: 指第一次从训练mode转换到评估 (eval) mode时
        # 它会执行一次滤波器的完整生成过程，并将最终的_filter和_bias通过register_buffer缓存起来
        # 之后再调用forward时，forward函数会发现自己处于评估模式，就会直接使用缓存好的self.filter，跳过所有生成步骤，从而大大加快推理速度
        elif self.training:
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights) # [outNum, tranNum, inNum, expand, sizeP, sizeP]

            Num = self.tranNum // self.expand

            # tempWList是个list, 长度为expand, 每个元素为六维张量[outNum, Num, inNum, expand, sizeP, sizeP]
            # 当为中间层时, tempWList长度为tranNum, 每个元素形状为[outNum, 1, inNum, expand, sizeP, sizeP]
            tempWList = [torch.cat([tempW[:, i * Num: (i + 1) * Num, :, -i:, :, :], tempW[:, i * Num: (i + 1) * Num, :, :-i, :, :]], dim=3) for i in range(self.expand)]
            tempW = torch.cat(tempWList, dim=1) # [outNum, tranNum, inNum, expand, sizeP, sizeP]

            _filter = torch.reshape(tempW, [self.outNum * self.tranNum, self.inNum * self.expand, self.sizeP, self.sizeP]) # [C_out, C_in, Kernel_H, Kernel_W]
            self.register_buffer('filter', _filter)

            if self.ifbias:
                _bias = self.c.repeat([1, 1, self.tranNum, 1]).reshape([1, self.outNum * self.tranNum, 1, 1]) # [1, outNum * tranNum, 1, 1]
                self.register_buffer('bias', _bias)

        return super().train(mode)







class Fconv_Up(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, ifIni=0, bias=True, Smooth=True, stride=2):
        super().__init__()

        if inP is None:
            inP = sizeP
        self.sizeP = sizeP
        self.inNum = inNum
        self.outNum = outNum
        self.tranNum = tranNum
        self.ifbias = bias
        if bias:
            self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1))
        else:
            self.register_parameter('c', None)
        if ifIni:
            expand = 1
        else:
            expand =  tranNum
        self.expand = expand

        self.stride = stride

        Basis, Rank, Weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth) # Basis: 四维张量[sizeP, sizeP, tranNum, Rank], Weight: 一维向量[Rank]

        self.register_buffer('Basis', Basis) # [sizeP, sizeP, tranNum, Rank]
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, Rank), requires_grad=True) # [outNum, inNum, expand, Rank]

        self.reset_parameters() # 初始化参数


    def reset_parameters(self) -> None: # 这也是Pytorch官方对Linear层的默认初始化方法，即Kaiming初始化
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.c, -bound, bound)


    # input: [Batch, C_in, H, W] = [Batch, inNum * expand, H, W]
    def forward(self, input):

        p = self.stride + self.sizeP - 2
        B, C, H, W = input.shape

        if self.training:
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights) # [outNum, tranNum, inNum, expand, sizeP, sizeP]

            Num = self.tranNum // self.expand # 若为输入层, Num = tranNum；若为中间层, Num = 1

            # 将tempW中每个tranNum通道，其中的expand个输入角度对应的滤波器进行排列，按照公式25的顺序排列好
            # tempWList是个list, 长度为expand, 每个元素为六维张量[outNum, Num, inNum, expand, sizeP, sizeP]
            # 当为中间层时, tempWList长度为tranNum, 每个元素形状为[outNum, 1, inNum, expand, sizeP, sizeP]
            # 当为输入层时, tempWList长度为1, 元素形状为[outNum, tranNum, inNum, 1(expand), sizeP, sizeP]
            tempWList = [torch.cat([tempW[:, i * Num: (i + 1) * Num, :, -i:, :, :], tempW[:, i * Num: (i + 1) * Num, :, :-i, :, :]], dim=3) for i in range(self.expand)]

            tempW = torch.cat(tempWList, dim=1) # [outNum, tranNum, inNum, expand, sizeP, sizeP]

            if p % 2 == 0:
                # 调整维度以匹配 conv_transpose2d 的要求 [in_channels, out_channels, kH, kW]
                tempW_permuted = tempW.permute(2, 3, 0, 1, 4, 5)  # from [o,t,i,e,h,w] to [i,e,o,t,h,w]
                _filter = torch.reshape(tempW_permuted,[self.inNum * self.expand, self.outNum * self.tranNum, self.sizeP, self.sizeP])
            else:
                _filter = torch.reshape(tempW,[self.outNum * self.tranNum, self.inNum * self.expand, self.sizeP, self.sizeP])

            if self.ifbias:
                _bias = self.c.repeat([1, 1, self.tranNum, 1]).reshape([1, self.outNum * self.tranNum, 1, 1]) # [1, outNum * tranNum, 1, 1]
                self.register_buffer('bias', _bias)
        else:
            _filter = self.filter
            if self.ifbias:
                _bias = self.bias


        if p % 2 == 0:
            if self.sizeP >= self.stride:
                output = F.conv_transpose2d(input, _filter, stride=self.stride, padding=(self.sizeP - self.stride) // 2, dilation=1, groups=1) # [Batch, outNum * tranNum, H_new, W_new]
            else:
                # 速度更快
                outpad = (self.stride - self.sizeP) // 2
                temp_output = F.conv_transpose2d(input, _filter, stride=self.stride, padding=0, dilation=1, groups=1)
                output = F.pad(temp_output, (outpad, outpad, outpad, outpad), mode="constant", value=0)

                # 更省内存
                # weight = torch.ones(C, 1, 1, 1, device=input.device, dtype=input.dtype)
                # padded_input = F.conv_transpose2d(input, weight, stride=self.stride, padding=0, groups=C)
                # output = F.conv2d(padded_input, _filter, padding=p // 2, dilation=1, groups=1)

        else:
            p1, p2 = (p - 1) // 2, (p + 1) // 2

            # 元素中间插入stride-1行、列零
            weight = torch.ones(C, 1, 1, 1, device=input.device, dtype=input.dtype)
            padded_input = F.conv_transpose2d(input, weight, stride=self.stride, padding=0, groups=C)

            # 周围旋转等变padding

            # 旧方法
            final_input = torch.zeros([B, C, self.stride * H + self.sizeP - 1, self.stride * W + self.sizeP - 1], device=input.device, dtype=input.dtype)

            final_input[:, 0::4, :, :] = F.pad(padded_input[:, 0::4, :, :], (p1, p2, p1, p2), mode="constant", value=0)
            final_input[:, 1::4, :, :] = F.pad(padded_input[:, 1::4, :, :], (p1, p2, p2, p1), mode="constant", value=0)
            final_input[:, 2::4, :, :] = F.pad(padded_input[:, 2::4, :, :], (p2, p1, p2, p1), mode="constant", value=0)
            final_input[:, 3::4, :, :] = F.pad(padded_input[:, 3::4, :, :], (p2, p1, p1, p2), mode="constant", value=0)


            output = F.conv2d(final_input, _filter, dilation=1, groups=1)


        if self.ifbias:
            output = output + _bias # [Batch, outNum * tranNum, H_new, W_new]

        return output

    # 重写nn.Module的train方法, 实现缓存的优化
    def train(self, mode=True):
        # 在训练mode时, 由于每次反向传播之后self.weights都会更新, 因此每次forward调用就要根据最新的weights来重新计算新的卷积核_filter
        # 如果存在之前缓存的filter和bias, 就删掉, 强制forward函数进行重新计算
        if mode:
            if hasattr(self, 'filter'):
                del self.filter
                if self.ifbias:
                    del self.bias

        # mode=False && self.training=True: 指第一次从训练mode转换到评估 (eval) mode时
        # 它会执行一次滤波器的完整生成过程，并将最终的_filter和_bias通过register_buffer缓存起来
        # 之后再调用forward时，forward函数会发现自己处于评估模式，就会直接使用缓存好的self.filter，跳过所有生成步骤，从而大大加快推理速度
        elif self.training:
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights) # [outNum, tranNum, inNum, expand, sizeP, sizeP]

            Num = self.tranNum // self.expand

            # tempWList是个list, 长度为expand, 每个元素为六维张量[outNum, Num, inNum, expand, sizeP, sizeP]
            # 当为中间层时, tempWList长度为tranNum, 每个元素形状为[outNum, 1, inNum, expand, sizeP, sizeP]
            tempWList = [torch.cat([tempW[:, i * Num: (i + 1) * Num, :, -i:, :, :], tempW[:, i * Num: (i + 1) * Num, :, :-i, :, :]], dim=3) for i in range(self.expand)]
            tempW = torch.cat(tempWList, dim=1) # [outNum, tranNum, inNum, expand, sizeP, sizeP]

            p = self.stride + self.sizeP - 2

            if p % 2 == 0:
                # 调整维度以匹配 conv_transpose2d 的要求 [in_channels, out_channels, kH, kW]
                tempW_permuted = tempW.permute(2, 3, 0, 1, 4, 5)  # from [o,t,i,e,h,w] to [i,e,o,t,h,w]
                _filter = torch.reshape(tempW_permuted,[self.inNum * self.expand, self.outNum * self.tranNum, self.sizeP, self.sizeP])
            else:
                _filter = torch.reshape(tempW,[self.outNum * self.tranNum, self.inNum * self.expand, self.sizeP, self.sizeP])

            self.register_buffer('filter', _filter)

            if self.ifbias:
                _bias = self.c.repeat([1, 1, self.tranNum, 1]).reshape([1, self.outNum * self.tranNum, 1, 1]) # [1, outNum * tranNum, 1, 1]
                self.register_buffer('bias', _bias)

        return super().train(mode)


class Fconv_1X1_out(nn.Module):
    def __init__(self, inNum, outNum, tranNum=4, bias=True, padding=0, padding_mode="zeros"):
        """
        修正后的等变 1x1 卷积输出层。
        通过强制权重在 tranNum 维度上共享，实现旋转不变的特征聚合（输出随空间旋转）。
        """
        super(Fconv_1X1_out, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.padding = padding
        self.padding_mode = padding_mode
        self.ifbias = bias

        # 【关键修改】
        # 权重维度改为 [outNum, inNum, 1, 1]
        # 这里不包含 tranNum 维度，意味着我们强制让所有方向共享同一个权重 W_i
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, 1, 1), requires_grad=True)

        # Bias 设置
        if bias:
            self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=True)
        else:
            self.register_parameter('c', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.c, -bound, bound)

    def forward(self, input):
        # input shape: [Batch, inNum * tranNum, H, W]

        # 1. 准备卷积核
        if self.training:
            # 构造共享权重的 Filter
            # 原始权重: [outNum, inNum, 1, 1]
            # 扩展目标: [outNum, inNum * tranNum, 1, 1]
            # 逻辑：对于每一个输出通道，每一个输入特征 i 的所有方向 t，都使用相同的权重

            tempW = self.weights.unsqueeze(2)  # [outNum, inNum, 1, 1, 1]
            tempW = tempW.repeat(1, 1, self.tranNum, 1, 1)  # [outNum, inNum, tranNum, 1, 1]
            _filter = tempW.reshape(self.outNum, self.inNum * self.tranNum, 1, 1)

            if self.ifbias:
                _bias = self.c
                self.register_buffer("bias", _bias)
        else:
            # 评估模式使用缓存
            _filter = getattr(self, 'filter', None)
            # 如果是 eval 模式第一次运行且没有缓存（edge case），则现场计算
            if _filter is None:
                tempW = self.weights.unsqueeze(2).repeat(1, 1, self.tranNum, 1, 1)
                _filter = tempW.reshape(self.outNum, self.inNum * self.tranNum, 1, 1)

            if self.ifbias:
                _bias = getattr(self, 'bias', self.c)

        # 2. 处理 Padding
        if self.padding_mode == 'zeros':
            padded_input = input
            conv_padding = self.padding
        else:
            if isinstance(self.padding, int):
                pad_arg = (self.padding, self.padding, self.padding, self.padding)
            elif isinstance(self.padding, tuple):
                pad_arg = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            else:
                raise ValueError("Padding must be int or tuple")

            padded_input = F.pad(input, pad_arg, mode=self.padding_mode)
            conv_padding = 0

        # 3. 执行卷积
        output = F.conv2d(padded_input,
                          _filter,
                          padding=conv_padding,
                          dilation=1,
                          groups=1)

        # 4. 加上 Bias
        if self.ifbias:
            output = output + _bias

        return output

    def train(self, mode=True):
        if mode:
            # 切换回训练模式，清除缓存
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias:
                    del self.bias
        elif self.training:
            # 切换到评估模式，预计算并缓存 Filter
            # 这样推理时无需重复执行 repeat 和 reshape 操作
            tempW = self.weights.unsqueeze(2).repeat(1, 1, self.tranNum, 1, 1)
            _filter = tempW.reshape(self.outNum, self.inNum * self.tranNum, 1, 1)
            self.register_buffer("filter", _filter)

            if self.ifbias:
                _bias = self.c
                self.register_buffer("bias", _bias)

        return super(Fconv_1X1_out, self).train(mode)