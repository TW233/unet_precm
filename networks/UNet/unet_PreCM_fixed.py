import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.UNet.PreCM_clean import PreCM1, PreCM2, PreCM3


class GroupyConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(GroupyConv1, self).__init__()
        self.gconv1 = PreCM1(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)


    def forward(self, x, output_shape):
        x = self.gconv1(x, output_shape)
        x = self.bn1(x)
        x = self.relu1(x)

        return x

class GroupyConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(GroupyConv2, self).__init__()
        self.gconv2 = PreCM2(in_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, output_shape):
        x = self.gconv2(x, output_shape)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class GroupyConv3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(GroupyConv3, self).__init__()
        self.gconv3 = PreCM3(in_channels, out_channels, kernel_size, stride, padding)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.gconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv1 = GroupyConv1(in_ch, out_ch, 3, 1, 1)
        self.conv2 = GroupyConv2(out_ch, out_ch, 3, 1, 1)

    def forward(self, x, output_shape):
        x = self.conv1(x, output_shape)
        x = self.conv2(x, output_shape)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.conv1 = GroupyConv2(in_ch, out_ch, 3, 2, 1)
        self.conv2 = GroupyConv2(out_ch, out_ch, 3, 1, 1)

    def forward(self, x, output_shape):
        x = self.conv1(x,  output_shape)
        x = self.conv2(x, output_shape)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2) # // 除以的结果向下取整

        self.conv3 = GroupyConv2(in_ch, out_ch, 3, 1, 1)

    def forward(self, x1, x2, output_shape):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv3(x, output_shape)
        return x


class finalConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(finalConv, self).__init__()
        self.finalconv = PreCM3(in_ch, out_ch, 3, 1, 1)
        # self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x, output):
        x = self.finalconv(x, output)
        # x = self.bn(x)
        return x


class unet_gconv(nn.Module):
    def __init__(self, in_channels=3, classes=2):
        super(unet_gconv, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes

        # self.inc = InConv(in_channels, 32)
        # self.down1 = Down(32, 64)
        # self.down2 = Down(64, 128)
        # self.down3 = Down(128, 256)
        # self.down4 = Down(256, 256)
        # self.up1 = Up(512, 128)
        # self.up2 = Up(256, 64)
        # self.up3 = Up(128, 32)
        # self.up4 = Up(64, 32)
        # self.outc = finalConv(32, classes)

        self.inc = InConv(in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 128)
        self.up1 = Up(256, 64)
        self.up2 = Up(128, 32)
        self.up3 = Up(64, 16)
        self.up4 = Up(32, 16)
        self.outc = finalConv(16, classes)

        self._init_weights()
    
    def _init_weights(self):
        # 1. 先对所有常规层进行 Kaiming 初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 2. 【关键修复】专门针对 PreCM 的 weight0 进行初始化
        # 必须使用 named_parameters 才能找到它们
        for name, param in self.named_parameters():
            if 'weight0' in name:
                # 使用 Kaiming Normal，修正方差
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                
                # 可选：如果你想尝试更保守的初始化（对应 PreCM 的 Summation 结构）
                # 可以尝试将 scale 缩小一半，防止方差爆炸。但在有 BN 的情况下 Kaiming 通常也行。
                # with torch.no_grad():
                #     param.mul_(0.5)

    def forward(self, x):
        x1 = self.inc(x, [448, 448])
        x2 = self.down1(x1, [224, 224])
        x3 = self.down2(x2, [112, 112])
        x4 = self.down3(x3, [56, 56])
        x5 = self.down4(x4, [28, 28])
        x = self.up1(x5, x4, [56, 56])
        x = self.up2(x, x3, [112, 112])
        x = self.up3(x, x2, [224, 224])
        x = self.up4(x, x1, [448, 448])
        x = self.outc(x, [448, 448])

        # x1 = self.inc(x, [256, 256])
        # x2 = self.down1(x1, [128, 128])
        # x3 = self.down2(x2, [64, 64])
        # x4 = self.down3(x3, [32, 32])
        # x5 = self.down4(x4, [16, 16])
        # x = self.up1(x5, x4, [32, 32])
        # x = self.up2(x, x3, [64, 64])
        # x = self.up3(x, x2, [128, 128])
        # x = self.up4(x, x1, [256, 256])
        # x = self.outc(x, [256, 256])

        return x

if __name__ == '__main__':
    device = torch.device('cuda:0')
    img = torch.rand(2, 3, 256, 256)
    # print(img)
    net = unet_gconv()
    for name, param in net.named_parameters():
        nn.init.normal_(param, -0.01, 0.02)
    output = net(img)
    # print(output)
    img1 = torch.rot90(img, dims=(-1, -2))
    output1 = net(img1)
    output1 = torch.rot90(output1, k=-1, dims=(-1, -2))
    difference = output1 - output
    print(difference)

