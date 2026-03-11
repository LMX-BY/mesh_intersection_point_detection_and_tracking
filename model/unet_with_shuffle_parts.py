""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import model.act_fn_config as af


def shuffle_chnls(x, groups=2):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()
    if chnls % groups:
        return x
    chnls_per_group = chnls // groups
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)
    return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    # (卷积 => [批归一化] => ReLU) * 2
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            af.act_fn,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            af.act_fn
        )

    # x，即输入特征图
    def forward(self, x):
        return self.double_conv(x)


class ShuffleConvWithSameInAndOutChnls(nn.Module):

    def __init__(self, in_channels, groups=2, c_ratio=0.5):
        super().__init__()
        self.groups = groups
        self.l_chnls = int(in_channels * c_ratio)
        self.r_chnls = in_channels - self.l_chnls
        self.conv_r1 = nn.Conv2d(self.r_chnls, self.r_chnls, kernel_size=1, stride=1,
                                 padding=0, groups=1, bias=False)
        self.conv_r2 = nn.Conv2d(self.r_chnls, self.r_chnls, kernel_size=3, stride=1,
                                                  padding=1, groups=self.r_chnls, bias=False)
        self.conv_r3 = nn.Conv2d(self.r_chnls, self.r_chnls, kernel_size=1, stride=1,
                                 padding=0, groups=1, bias=False)
        self.bn_r = nn.BatchNorm2d(self.r_chnls)
        self.af = af.act_fn

    def forward(self, x):
        x_b_2_l = x[:, :self.l_chnls, :, :]
        x_b_2_r = x[:, self.l_chnls:, :, :]

        out_b_2_r = self.conv_r1(x_b_2_r)
        out_b_2_r = self.bn_r(out_b_2_r)
        out_b_2_r = self.af(out_b_2_r)
        out_b_2_r = self.conv_r2(out_b_2_r)
        out_b_2_r = self.bn_r(out_b_2_r)
        out_b_2_r = self.conv_r3(out_b_2_r)
        out_b_2_r = self.bn_r(out_b_2_r)
        out_b_2_r = self.af(out_b_2_r)

        # concatenate
        out = torch.cat((x_b_2_l, out_b_2_r), 1)
        return shuffle_chnls(out, self.groups)


class ShuffleConvWithSameInAndOutChnls2(nn.Module):

    def __init__(self, in_channels, groups=2, c_ratio=0.5):
        super().__init__()
        self.groups = groups
        self.l_chnls = int(in_channels * c_ratio)
        self.r_chnls = in_channels - self.l_chnls
        self.conv_r1 = nn.Conv2d(self.r_chnls, self.r_chnls, kernel_size=3, stride=1,
                                                  padding=1, groups=self.r_chnls, bias=False)
        self.bn_r = nn.BatchNorm2d(self.r_chnls)
        self.af = af.act_fn

    def forward(self, x):
        x_b_2_l = x[:, :self.l_chnls, :, :]
        x_b_2_r = x[:, self.l_chnls:, :, :]

        out_b_2_r = self.conv_r1(x_b_2_r)
        out_b_2_r = self.bn_r(out_b_2_r)
        out_b_2_r = self.af(out_b_2_r)

        # concatenate
        out = torch.cat((x_b_2_l, out_b_2_r), 1)
        return shuffle_chnls(out, self.groups)


# 下采样
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    # 最大池化下采样，然后双卷积
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DownWithSingleShuffleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    # 最大池化下采样，然后双卷积
    def __init__(self, in_channels, groups=2):
        super().__init__()
        self.groups = groups
        self.dwconv_l1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2,
                                   padding=1, groups=in_channels, bias=False)
        self.bn_l = nn.BatchNorm2d(in_channels)
        self.conv_l2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1,
                                 padding=0, groups=1, bias=False)
        self.af = af.act_fn

        self.conv_r1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1,
                                 padding=0, groups=1, bias=False)
        self.bn_r = nn.BatchNorm2d(in_channels)

        self.dwconv_r2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2,
                                   padding=1, groups=in_channels, bias=False)
        self.conv_r3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1,
                                 padding=0, groups=1, bias=False)

    def forward(self, x):
        # left path
        out_l = self.dwconv_l1(x)
        out_l = self.bn_l(out_l)
        out_l = self.conv_l2(out_l)
        out_l = self.bn_l(out_l)
        out_l = self.af(out_l)

        # right path
        out_r = self.conv_r1(x)
        out_r = self.bn_r(out_r)
        out_r = self.af(out_r)
        out_r = self.dwconv_r2(out_r)
        out_r = self.bn_r(out_r)
        out_r = self.conv_r3(out_r)
        out_r = self.bn_r(out_r)
        out_r = self.af(out_r)

        # concatenate
        out = torch.cat((out_l, out_r), 1)
        return shuffle_chnls(out, self.groups)


class DownWithSingleShuffleConv2(nn.Module):
    """Downscaling with maxpool then double conv"""

    # 最大池化下采样，然后双卷积
    def __init__(self, in_channels, groups=2):
        super().__init__()
        self.groups = groups
        self.dwconv_l1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2,
                                   padding=1, groups=1, bias=False)

        self.dwconv_r2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2,
                                   padding=1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.af = af.act_fn

    def forward(self, x):
        # left path
        out_l = self.dwconv_l1(x)
        out_l = self.bn(out_l)
        out_l = self.af(out_l)

        # right path
        out_r = self.dwconv_r2(x)
        out_r = self.bn(out_r)
        out_r = self.af(out_r)

        # concatenate
        out = torch.cat((out_l, out_r), 1)
        return shuffle_chnls(out, self.groups)


class DownWithDoubleShuffleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    # 最大池化下采样，然后双卷积
    def __init__(self, in_channels, out_channels, groups=2, c_ratio=0.5):
        super().__init__()
        assert in_channels * 2 == out_channels
        self.out_channels = out_channels
        # 通道数扩大两倍
        self.down_s_conv1 = DownWithSingleShuffleConv2(in_channels, groups)
        self.shuffle_conv1 = ShuffleConvWithSameInAndOutChnls2(out_channels, groups, c_ratio)

    def forward(self, x):
        x_out = self.down_s_conv1(x)
        x_out = self.shuffle_conv1(x_out)
        return x_out


# 上采样
class UpWithDoubleShuffleConv(nn.Module):
    """Upscaling then double conv"""

    # 上采样，然后双卷积
    def __init__(self, in_channels, out_channels, groups=2, c_ratio=0.5, bilinear=True):
        super().__init__()
        assert in_channels // 2 == out_channels
        self.groups = groups
        self.bilinear = bilinear
        # 如果使用双线性插值，则使用普通的卷积来减少通道数
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 转置卷积
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_cvt_c1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                     padding=0, groups=1, bias=False)
        self.conv_cvt_c2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                     padding=0, groups=1, bias=False)
        self.af = af.act_fn
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv_shuffle_1 = ShuffleConvWithSameInAndOutChnls2(out_channels, groups, c_ratio)
        self.conv_shuffle_2 = ShuffleConvWithSameInAndOutChnls2(out_channels, groups, c_ratio)

    def forward(self, x1, x2):  # x1通过上采样操作得到，x2是来自下采样路径的特征图
        if self.bilinear:
            x1_up = self.up(x1)
            x1_up = self.conv_cvt_c1(x1_up)
            x1_up = self.bn(x1_up)
            x1_up = self.af(x1_up)
        else:
            x1_up = self.up(x1)
        # # 输入是CHW格式 尺寸对齐
        # diffY = x2.size()[2] - x1.size()[2]  # x1和x2在高度上的尺寸差异
        # diffX = x2.size()[3] - x1.size()[3]  # x1和x2在宽度上的尺寸差异
        #
        # # 差异值(diffY和diffX)用于对x1进行填充操作，填充方式为在左、右、上、下分别填充 diffX // 2、diffX - diffX // 2、diffY // 2、diffY - diffY // 2。
        # x1_up = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # 两种特征图拼接
        x_cat = shuffle_chnls(torch.cat([x2, x1_up], dim=1), self.groups)
        #x_cat = torch.cat([x2, x1_up], dim=1)
        x_cat = self.conv_cvt_c2(x_cat)
        x_cat = self.bn(x_cat)
        x_cat = self.af(x_cat)
        x_cat = self.conv_shuffle_1(x_cat)
        x_cat = self.conv_shuffle_2(x_cat)
        return x_cat


class Up(nn.Module):
    """Upscaling then double conv"""

    # 上采样，然后双卷积
    def __init__(self, in_channels, out_channels, bilinear=True, groups=2):
        super().__init__()
        assert in_channels // 2 == out_channels
        self.bilinear = bilinear
        self.groups = groups
        # 如果使用双线性插值，则使用普通的卷积来减少通道数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 转置卷积
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_cvt_c1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                     padding=0, groups=1, bias=False)
        self.af = af.act_fn
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):  # x1通过上采样操作得到，x2是来自下采样路径的特征图
        if self.bilinear:
            x1_up = self.up(x1)
            x1_up = self.conv_cvt_c1(x1_up)
            x1_up = self.bn(x1_up)
            x1_up = self.af(x1_up)
        else:
            x1_up = self.up(x1)
        # # 输入是CHW格式 尺寸对齐
        # diffY = x2.size()[2] - x1.size()[2]  # x1和x2在高度上的尺寸差异
        # diffX = x2.size()[3] - x1.size()[3]  # x1和x2在宽度上的尺寸差异
        #
        # # 差异值(diffY和diffX)用于对x1进行填充操作，填充方式为在左、右、上、下分别填充 diffX // 2、diffX - diffX // 2、diffY // 2、diffY - diffY // 2。
        # x1_up = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # 两种特征图拼接
        x_cat = shuffle_chnls(torch.cat([x2, x1_up], dim=1), self.groups)

        # 将拼接后的特征图 x 输入 DoubleConv 模块进行双重卷积，得到最终的输出特征图
        return self.conv(x_cat)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=(1, 1), padding=1),
            af.act_fn,
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=(1, 1), padding=1),
            af.act_fn,
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)  # 输出形状为 (batch_size, out_channels, height, width) 的特征图。
