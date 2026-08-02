import math
import torch
import torch.nn as nn
from .config import FLAGS
from .ops import ResnetConv2d, ResnetBatchNorm2d, ResnetLinear, ActQuant, make_divisible

class Block(nn.Module):
    def __init__(self, inp, outp, stride):
        super(Block, self).__init__()
        assert stride in [1, 2]

        midp = outp // 4
        self.conv1 = ResnetConv2d(inp, midp, 1, 1, 0, bias=False)
        self.bn1 = ResnetBatchNorm2d(midp)
        self.act1 = ActQuant()

        self.conv2 = ResnetConv2d(midp, midp, 3, stride, 1, bias=False)
        self.bn2 = ResnetBatchNorm2d(midp)
        self.act2 = ActQuant()

        self.conv3 = ResnetConv2d(midp, outp, 1, 1, 0, bias=False)
        self.bn3 = ResnetBatchNorm2d(outp)

        self.relu = nn.ReLU(inplace=True)

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut_conv = ResnetConv2d(inp, outp, 1, stride=stride, bias=False)
            self.shortcut_bn = ResnetBatchNorm2d(outp)

        self.post_act = ActQuant()

    def forward(self, x):
        out = self.act1(self.relu(self.bn1(self.conv1(x))))
        out = self.act2(self.relu(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))

        if self.residual_connection:
            res = x
        else:
            res = self.shortcut_bn(self.shortcut_conv(x))

        out = self.relu(out + res)
        out = self.post_act(out)
        return out


class Model(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super(Model, self).__init__()
        self.features = []

        self.block_setting_dict = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        self.block_setting = self.block_setting_dict[FLAGS.depth]
        feats = [64, 128, 256, 512]
        max_width = max(FLAGS.width_mult_range)
        channels = make_divisible(64 * max_width, FLAGS.width_divisor)

        if FLAGS.dataset == "cifar10":
            self.features.append(nn.Sequential(
                ResnetConv2d(3, channels, 3, 1, 1, bias=False, is_stem=True),
                ResnetBatchNorm2d(channels),
                nn.ReLU(inplace=True),
                ActQuant(),
            ))
        else:
            assert input_size % 32 == 0
            self.features.append(nn.Sequential(
                ResnetConv2d(3, channels, 7, 2, 3, bias=False, is_stem=True),
                ResnetBatchNorm2d(channels),
                nn.ReLU(inplace=True),
                ActQuant(),
                nn.MaxPool2d(3, 2, 1),
            ))

        for stage_id, n in enumerate(self.block_setting):
            outp = make_divisible(feats[stage_id] * max_width * 4, FLAGS.width_divisor)
            for i in range(n):
                stride = 2 if (i == 0 and stage_id != 0) else 1
                self.features.append(Block(channels, outp, stride))
                channels = outp

        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*self.features)

        self.outp = channels
        self.classifier = nn.Sequential(ResnetLinear(self.outp, num_classes))
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        last_dim = x.size()[1]
        x = x.view(-1, last_dim)
        x = self.classifier(x)
        return x

    def set_width(self, width_mult):
        for m in self.modules():
            if hasattr(m, 'width_mult'):
                m.width_mult = width_mult

    def set_bit_width(self, bit_width):
        for m in self.modules():
            if hasattr(m, 'bit_width'):
                m.bit_width = bit_width

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.weight.size(0) * m.kernel_size[0] * m.kernel_size[1]
                nn.init.normal_(m.weight, 0, math.sqrt(2. / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def set_model_width(m, width_mult):
    (m.module if hasattr(m, 'module') else m).set_width(width_mult)


def set_model_bit_width(m, bit_width):
    (m.module if hasattr(m, 'module') else m).set_bit_width(bit_width)
