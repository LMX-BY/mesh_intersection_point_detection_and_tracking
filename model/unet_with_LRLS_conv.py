""" Full assembly of the parts to form the complete network """

from model.LRLS_conv import *

class UNetWithLRLSConv(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetWithLRLSConv, self).__init__()
        # 初始化UNet类的属性
        self.n_channels = n_channels  # 输入图像的通道数
        self.n_classes = n_classes  # 分类的类别数
        self.bilinear = bilinear  # 是否使用双线性插值的上采样

        self.inc = (DoubleLRLSConv(n_channels, 64))  # 输入模块
        # 下采样模块
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        # 为了在定义 down4 模块时，根据是否使用双线性插值来确定输出通道数的值。如果使用双线性插值，则输出通道数为 1024 // 2 = 512
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        # 上采样模块
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))  # 输出模块

    # 执行
    def forward(self, x):
        # 类里面的方法继承自 nn.Module，forward就不用显式的调用
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    # 使用checkpoint对输入模块进行优化
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def set_phase_train(self):
        self.train()
        self.phase = 'train'

    def set_phase_eval(self):
        self.eval()
        self.phase = 'eval'

