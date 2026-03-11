import torch
import torch.nn as nn
import torch.nn.functional as F
import model.act_fn_config as af


class LSKA(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), groups=dim,
                                            dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=dim,
                                            dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 9), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1, 1), padding=(9, 0), groups=dim,
                                            dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=dim,
                                            dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1, 1), padding=(0, 18), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1, 1), padding=(18, 0), groups=dim,
                                            dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 24), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1, 1), padding=(24, 0), groups=dim,
                                            dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn


class DWLRLSConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, k_size, bias=False):
        super().__init__()
        self.cov1 = nn.Conv2d(input_channels, output_channels, kernel_size=(1, k_size), stride=(1, 1),
                              padding=(0, (k_size - 1) // 2), bias=bias, groups=input_channels)
        self.cov2 = nn.Conv2d(output_channels, output_channels, kernel_size=(k_size, 1), stride=(1, 1),
                              padding=((k_size - 1) // 2, 0), groups=output_channels, bias=bias)
        self.cov3 = nn.Conv2d(output_channels, output_channels, 1)

    def forward(self, x):
        x1 = self.cov1(x)
        x2 = self.cov2(x1)
        x3 = self.cov3(x2)
        return x3


# 两个横向卷积
class DoubleDWLRLSConvWithShortCut(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, k_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.cov1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act_fn = af.act_fn
        self.cov2 = DWLRLSConv2d(mid_channels, out_channels, k_size, False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.cov_short_cut = nn.Conv2d(mid_channels, out_channels, 1)

    # x，即输入特征图
    def forward(self, x):
        y = self.cov1(x)
        y = self.bn1(y)
        y = self.self.act_fn(y)
        shout_cut_y = self.cov_short_cut(y)
        y = self.cov2(y)
        y = self.bn2(y)
        out = shout_cut_y + y
        out = self.self.act_fn(out)
        return out


# 下采样
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    # 最大池化下采样，然后双卷积
    def __init__(self, in_channels, out_channels, k_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleDWLRLSConvWithShortCut(in_channels, out_channels, k_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    # 上采样，然后双卷积
    def __init__(self, in_channels, out_channels, k_size, bilinear=True):
        super().__init__()

        # 如果使用双线性插值，则使用普通的卷积来减少通道数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleDWLRLSConvWithShortCut(in_channels, out_channels, k_size, in_channels // 2)
        # 转置卷积
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleDWLRLSConvWithShortCut(in_channels, out_channels, k_size)

    def forward(self, x1, x2):  # x1通过上采样操作得到，x2是来自下采样路径的特征图
        x1 = self.up(x1)
        # 输入是CHW格式 尺寸对齐
        diffY = x2.size()[2] - x1.size()[2]  # x1和x2在高度上的尺寸差异
        diffX = x2.size()[3] - x1.size()[3]  # x1和x2在宽度上的尺寸差异

        # 差异值(diffY和diffX)用于对x1进行填充操作，填充方式为在左、右、上、下分别填充 diffX // 2、diffX - diffX // 2、diffY // 2、diffY - diffY // 2。
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # 两种特征图拼接
        x = torch.cat([x2, x1], dim=1)

        # 将拼接后的特征图 x 输入 DoubleConv 模块进行双重卷积，得到最终的输出特征图
        return self.conv(x)

# 上采样
class Up_1(nn.Module):
    """Upscaling then double conv"""

    # 上采样，然后双卷积
    def __init__(self, in_channels, out_channels, k_size, bilinear=True):
        super().__init__()

        # 如果使用双线性插值，则使用普通的卷积来减少通道数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleDWLRLSConvWithShortCut(in_channels, out_channels, k_size, in_channels // 2)
        # 转置卷积
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels - 2, kernel_size=2, stride=2)
            self.conv = DoubleDWLRLSConvWithShortCut((in_channels - 2) * 2, out_channels, k_size)

    def forward(self, x1, x2):  # x1通过上采样操作得到，x2是来自下采样路径的特征图
        x1 = self.up(x1)
        # 输入是CHW格式 尺寸对齐
        diffY = x2.size()[2] - x1.size()[2]  # x1和x2在高度上的尺寸差异
        diffX = x2.size()[3] - x1.size()[3]  # x1和x2在宽度上的尺寸差异

        # 差异值(diffY和diffX)用于对x1进行填充操作，填充方式为在左、右、上、下分别填充 diffX // 2、diffX - diffX // 2、diffY // 2、diffY - diffY // 2。
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # 两种特征图拼接
        x = torch.cat([x2, x1], dim=1)

        # 将拼接后的特征图 x 输入 DoubleConv 模块进行双重卷积，得到最终的输出特征图
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_size):
        super(OutConv, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            DWLRLSConv2d(in_channels, in_channels, k_size),
            af.act_fn,
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=(1, 1), padding=1),
            af.act_fn,
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)  # 输出形状为 (batch_size, out_channels, height, width) 的特征图。
