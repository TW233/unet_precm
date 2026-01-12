import torch
import torch.nn as nn
import torch.nn.functional as F

class PreCM1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups=1, bias=0):
        super(PreCM1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        
        weight_tensor = torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size).float()
        self.weight0 = nn.Parameter(weight_tensor)
        nn.init.kaiming_normal_(self.weight0, mode='fan_out', nonlinearity='relu') # 使用 Kaiming 初始化

    def forward(self, input, output_shape):
        ho, wo = output_shape[0], output_shape[1]
        b, c, h, w = input.shape
        pab = (ho - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - h
        prl = (wo - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - w
        pb = int(pab // 2); pl = int(prl // 2)
        pa = pab - pb; pr = prl - pl
        padding = (pa, pb, pl, pr)
        
        # 旋转 0, 90, 180, 270 并拼接
        input = torch.cat([input,
                           torch.rot90(input, k=-1, dims=(2, 3)),
                           torch.rot90(input, k=-2, dims=(2, 3)),
                           torch.rot90(input, k=-3, dims=(2, 3))], dim=0)
        return F.conv2d(F.pad(input, padding), weight=self.weight0, bias=None, stride=self.stride, groups=self.groups)


class PreCM2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, dilation=1, groups=1):
        super(PreCM2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        # 4倍输出通道
        weight_tensor0 = torch.Tensor(4 * out_channels, in_channels // groups, kernel_size, kernel_size).float()
        self.weight0 = nn.Parameter(weight_tensor0)
        nn.init.kaiming_normal_(self.weight0, mode='fan_out', nonlinearity='relu')

    def forward(self, input, output_shape):
        ho, wo = output_shape[0], output_shape[1]
        b, c, h, w = input.shape
        pab = (ho - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - h
        prl = (wo - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - w
        pb = int(pab // 2); pl = int(prl // 2)
        pa = pab - pb; pr = prl - pl
        padding = (pa, pb, pl, pr)
        
        out2 = F.conv2d(F.pad(input, padding), weight=self.weight0, bias=None, stride=self.stride, dilation=self.dilation, groups=self.groups)
        
        batch = b // 4
        oc = self.out_channels
        out2list = []
        # PreCM 核心逻辑：特征图旋转融合
        for i in range(4):
            out2list.append(
                torch.rot90(out2[0 * batch: 1 * batch, (i - 0) % 4 * oc: (i - 0) % 4 * oc + oc], k=(-i + 0) % 4, dims=(2, 3)) + \
                torch.rot90(out2[1 * batch: 2 * batch, (i - 1) % 4 * oc: (i - 1) % 4 * oc + oc], k=(-i + 1) % 4, dims=(2, 3)) + \
                torch.rot90(out2[2 * batch: 3 * batch, (i - 2) % 4 * oc: (i - 2) % 4 * oc + oc], k=(-i + 2) % 4, dims=(2, 3)) + \
                torch.rot90(out2[3 * batch: 4 * batch, (i - 3) % 4 * oc: (i - 3) % 4 * oc + oc], k=(-i + 3) % 4, dims=(2, 3))
            )
        return torch.cat(out2list, dim=0)


class PreCM3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, bias=True, groups=1): # bias 默认为 True
        super(PreCM3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        weight_tensor = torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size).float()
        self.weight0 = nn.Parameter(weight_tensor)
        nn.init.kaiming_normal_(self.weight0, mode='fan_out', nonlinearity='relu')
        
        # 【修正】添加 bias 参数
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, output_shape):
        ho, wo = output_shape[0], output_shape[1]
        b, c, h, w = input.shape
        pab = (ho - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - h
        prl = (wo - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - w
        pb = int(pab // 2); pl = int(prl // 2)
        pa = pab - pb; pr = prl - pl
        padding = (pa, pb, pl, pr)
        batch = b // 4
        
        # 这里的 bias 暂时填 None，我们在最后加
        out3 = F.conv2d(F.pad(input, padding), weight=self.weight0, bias=None, stride=self.stride, groups=self.groups)
        
        # 融合 4 个方向的特征
        out = torch.rot90(out3[0 * batch: 1 * batch], k=0, dims=(2, 3)) + \
              torch.rot90(out3[1 * batch: 2 * batch], k=1, dims=(2, 3)) + \
              torch.rot90(out3[2 * batch: 3 * batch], k=2, dims=(2, 3)) + \
              torch.rot90(out3[3 * batch: 4 * batch], k=3, dims=(2, 3))
              
        # 【修正】手动加上 bias
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
            
        return out