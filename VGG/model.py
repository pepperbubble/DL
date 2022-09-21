import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, class_num=1000):
        super(VGG, self).__init__()
        self.feature = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(7 * 7 * 512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, class_num)
        )

    def forward(self, x):
        # x:N * 3 * 224 * 224
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)  # 不对batch展平
        x = self.classifier(x)
        return x


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_feature(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)  # *表示非关键字参数


def vgg(mode_name="vgg16", **kwargs): #**kwargs表示可变长度字典
    cfg = cfgs[mode_name]
    model = VGG(make_feature(cfg), **kwargs)
    return model
