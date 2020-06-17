from collections import OrderedDict
from sacred import Ingredient

import torch
import torch.nn as nn
import torch.nn.functional as F
from baal.active.heuristics import BALD, Variance, Entropy

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, dropout=False):
        super(DoubleConv, self).__init__()

        modules_list = [
            ('conv1', nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(True)),
        ]

        if dropout:
            modules_list.append(('dropout', nn.Dropout(0.2)))

        layers = OrderedDict(modules_list)

        for name, module in layers.items():
            self.add_module(name, module)

class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, base_width=16, dropout=False):
        super(UNet, self).__init__()

        self.base_width = base_width

        self.init_conv = DoubleConv(in_channels, self.base_width, stride=1, dropout=dropout)
        self.down_convs = nn.ModuleList([
            DoubleConv(self.base_width, self.base_width * 2, stride=2, dropout=dropout),
            DoubleConv(self.base_width * 2, self.base_width * 4, stride=2, dropout=dropout),
            DoubleConv(self.base_width * 4, self.base_width * 8, stride=2, dropout=dropout),
            DoubleConv(self.base_width * 8, self.base_width * 16, stride=2, dropout=dropout),
            DoubleConv(self.base_width * 16, self.base_width * 32, stride=2, dropout=dropout),
        ])

        self.upsample_convs = nn.ModuleList([
            DoubleConv(self.base_width * 32, self.base_width * 32),
            DoubleConv(self.base_width * 16, self.base_width * 16),
            DoubleConv(self.base_width * 8, self.base_width * 8),
            DoubleConv(self.base_width * 4, self.base_width * 4),
            DoubleConv(self.base_width * 2, self.base_width * 2),
        ])

        self.up_convs = nn.ModuleList([
            DoubleConv(self.base_width * 48, self.base_width * 16, dropout=dropout),
            DoubleConv(self.base_width * 24, self.base_width * 8, dropout=dropout),
            DoubleConv(self.base_width * 12, self.base_width * 4, dropout=dropout),
            DoubleConv(self.base_width * 6, self.base_width * 2, dropout=dropout),
            DoubleConv(self.base_width * 3, self.base_width, dropout=dropout),
        ])

        self.end_conv = nn.Conv2d(self.base_width, n_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = [(x.shape[2:], self.init_conv(x))]
        for i, module in enumerate(self.down_convs):
            x = module(down[i][1])
            down.append((x.shape[2:], x))

        for i, (upsample, module) in enumerate(zip(self.upsample_convs, self.up_convs)):
            upsampled = F.interpolate(upsample(x), size=down[-(i + 2)][0])
            x = module(torch.cat((upsampled, down[-(i + 2)][1]), 1))

        return self.end_conv(x)

from collections.abc import Sequence

def _stack_preds(out):
    if isinstance(out[0], Sequence):
        out = [torch.stack(ts, dim=-1) for ts in zip(*out)]
    else:
        out = torch.stack(out, dim=-1)
    return out

if __name__ == '__main__':
    unet = UNet()
    # print(unet)

    x = torch.rand(1, 3, 733, 427)

    with torch.no_grad():
        preds = [unet(x) for _ in range(20)]
        preds_stack = _stack_preds(preds)

        # heuristic = BALD()
        heuristic = Variance()
        # heuristic = Entropy()

        metric = heuristic(preds_stack)

        print('Input:', x.shape, '-> UNet(input):', unet(x).shape)