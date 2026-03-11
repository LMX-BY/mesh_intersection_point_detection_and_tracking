import torch.nn as nn

act_fn = None


def set_activate_fn(type):
    global act_fn
    leaky_relu = nn.LeakyReLU(0.2, inplace=True)
    relu = nn.ReLU(inplace=True)
    act_fn = None
    if type == 'relu':
        act_fn = relu
    if type == 'leaky_relu':
        act_fn = leaky_relu
    assert act_fn is not None

