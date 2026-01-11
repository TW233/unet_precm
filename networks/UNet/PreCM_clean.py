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
        
        # 修复：只保留实际使用的 Parameter，删除多余的 nn.Conv2d
        weight_tensor = torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size).float()
        self.weight0 = nn.Parameter(weight_tensor)
        
        # 初始化
        nn.init.normal_(self.weight0, -0.01, 0.02)

    def forward(self, input, output_shape):
        ho, wo = output_shape[0], output_shape[1]
        b, c, h, w = input.shape
        pab = (ho - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - h
        prl = (wo - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - w
        pb = int(pab // 2)
        pl = int(prl // 2)
        pa = pab - pb
        pr = prl - pl
        padding = (pa, pb, pl, pr)
        
        # 旋转扩充 Batch
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
        
        # 修复：只保留 self.weight0，删除 self.convtest
        # 注意：PreCM2 的权重输出通道是 4 * out_channels
        weight_tensor0 = torch.Tensor(4 * out_channels, in_channels // groups, kernel_size, kernel_size).float()
        self.weight0 = nn.Parameter(weight_tensor0)
        
        # 初始化
        nn.init.normal_(self.weight0, -0.01, 0.02)

    def forward(self, input, output_shape):
        ho, wo = output_shape[0], output_shape[1]
        b, c, h, w = input.shape
        pab = (ho - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - h
        prl = (wo - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - w
        pb = int(pab // 2)
        pl = int(prl // 2)
        pa = pab - pb
        pr = prl - pl
        padding = (pa, pb, pl, pr)
        
        out2 = F.conv2d(F.pad(input, padding), weight=self.weight0, bias=None, stride=self.stride, dilation=self.dilation, groups=self.groups)
        
        batch = b // 4
        oc = self.out_channels
        out2list = []
        for i in range(4):
            # 这里的切片和旋转逻辑保持不变，这是PreCM的核心群卷积逻辑
            out2list.append(
                torch.rot90(out2[0 * batch: 0 * batch + batch, (i - 0) % 4 * oc: (i - 0) % 4 * oc + oc, :, :], k=(-i + 0) % 4, dims=(2, 3)) + \
                torch.rot90(out2[1 * batch: 1 * batch + batch, (i - 1) % 4 * oc: (i - 1) % 4 * oc + oc, :, :], k=(-i + 1) % 4, dims=(2, 3)) + \
                torch.rot90(out2[2 * batch: 2 * batch + batch, (i - 2) % 4 * oc: (i - 2) % 4 * oc + oc, :, :], k=(-i + 2) % 4, dims=(2, 3)) + \
                torch.rot90(out2[3 * batch: 3 * batch + batch, (i - 3) % 4 * oc: (i - 3) % 4 * oc + oc, :, :], k=(-i + 3) % 4, dims=(2, 3))
            )
        return torch.cat(out2list, dim=0)


class PreCM3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, bias=0, groups=1):
        super(PreCM3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        # 修复：原代码 PreCM3 用的是 self.convtest.weight，所以我们删除 self.weight0，只保留 Conv2d
        # 为了保持一致性，我们这里干脆也改成用 Parameter，这样更清晰
        weight_tensor = torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size).float()
        self.weight0 = nn.Parameter(weight_tensor)
        
        # 初始化
        nn.init.normal_(self.weight0, -0.01, 0.02)

    def forward(self, input, output_shape):
        ho, wo = output_shape[0], output_shape[1]
        b, c, h, w = input.shape
        pab = (ho - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - h
        prl = (wo - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - w
        pb = int(pab // 2)
        pl = int(prl // 2)
        pa = pab - pb
        pr = prl - pl
        padding = (pa, pb, pl, pr)
        batch = b // 4
        
        # 使用 weight0
        out3 = F.conv2d(F.pad(input, padding), weight=self.weight0, bias=None, stride=self.stride, groups=self.groups)
        
        return torch.rot90(out3[0 * batch: 1 * batch], k=0, dims=(2, 3)) + \
                torch.rot90(out3[1 * batch: 2 * batch], k=1, dims=(2, 3)) + \
                torch.rot90(out3[2 * batch: 3 * batch], k=2, dims=(2, 3)) + \
                torch.rot90(out3[3 * batch: 4 * batch], k=3, dims=(2, 3))