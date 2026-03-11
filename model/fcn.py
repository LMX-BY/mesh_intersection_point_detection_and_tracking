import torch.nn as nn
import torch
import torch.nn.functional as F
from config import fcn_config


class FCNNet(nn.Module):
    def __init__(self, input_channels, config):
        super(FCNNet, self).__init__()
        self.model = nn.Sequential()
        self.input_channels = input_channels
        cur_output_channels = self.input_channels
        for a_layer_name in config:
            if isinstance(a_layer_name, int):
                self.model.append(nn.Conv2d(cur_output_channels, a_layer_name, kernel_size=5, stride=1, padding=2))
                cur_output_channels = a_layer_name
                continue

            if a_layer_name == fcn_config.relu:
                self.model.append(nn.ReLU(inplace=True))
                continue

            if a_layer_name == fcn_config.max_pooling:
                self.model.append(nn.MaxPool2d(2))
                continue

            if a_layer_name == fcn_config.up_sample:
                self.model.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                continue

    def forward(self, x):
        cur_output = x
        for a_module in self.model:
            cur_output = a_module(cur_output)
        return cur_output

    def set_phase_train(self):
        self.train()
        self.phase = 'train'


if __name__ == '__main__':
    test_fcn = FCNNet(3, fcn_config.fcn1_config)
    test_img = torch.randn(1, 3, 560, 224)
    test_output = test_fcn(test_img)
    pass
