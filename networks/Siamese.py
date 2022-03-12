import torch
import torch.nn as nn
from torch.nn import init
import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'DCCA', 'model_dict']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation))

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, width=1.0, in_channel=3, unet=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        self.unet = unet
        self._norm_layer = nn.BatchNorm2d

        # self.inplanes = max(int(64 * width), 64)
        self.inplanes = int(64 * width)
        self.base = int(64 * width)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer0 = nn.Sequential(
            #nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False), # 64
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channel, self.inplanes, kernel_size=3, stride=2, padding=0, bias=False),  # 32
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True))
        self.maxpool = nn.Sequential(nn.ReplicationPad2d(1), nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        #self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2,
        #                               dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, mode=0):
        if mode == 0:
            outputs = {}
            module_list = [self.layer0, self.maxpool, self.layer1, self.layer2, self.layer3]
            for idx in range(len(module_list)):
                x = module_list[idx](x)
                outputs['d{}'.format(idx + 1)] = x
            return outputs
        else:
            x = self.layer0(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x

class Siamese(nn.Module):
    def __init__(self, n_inputs=6, n_clasess=10):
        super(Siamese, self).__init__()

        self.S2 = n_inputs - 2
        self.encoder2 = ResNet(BasicBlock, [2, 2, 2, 2], in_channel=self.S2, width=0.5)
        self.encoder1 = ResNet(BasicBlock, [2, 2, 2, 2], in_channel=2, width=0.5)
        self.fc = nn.Linear(256, n_clasess)

    def forward(self, x, mode=1):
        if mode==1:
            l, ab = torch.split(x, [self.S2, 2], dim=1)
            fl = self.encoder2(l, mode=1)
            fab = self.encoder1(ab, mode=1)
            x = torch.cat((fl, fab), dim=1)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        else:
            l, ab = torch.split(x, [self.S2, 2], dim=1)
            fl = self.encoder2(l, mode=mode)
            fab = self.encoder1(ab, mode=mode)
            return fl, fab


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)



model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
}


if __name__ == '__main__':

    model = resnet18(width=0.5)
    data = torch.randn(2, 3, 224, 224)
    out = model(data)
    print(out.shape)
