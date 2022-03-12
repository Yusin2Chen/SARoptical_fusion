import torch
from torch.nn import init
import torch.nn as nn
import math
from utils.losses import HardNegtive_loss

__all__ = ['ResNet', 'ResUnet']


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class MLPHead0(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead0, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        shape = x.shape
        x = x.view(-1, 256) #original 128
        out_put = self.net(x)
        out_put = out_put.reshape(shape)
        out_put = out_put.permute(0, 3, 1, 2)
        return out_put

class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mlp_hidden_size, kernel_size=1, stride=1),
            #nn.BatchNorm2d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_hidden_size, projection_size, kernel_size=1, stride=1)
        )

    def forward(self, x):
        out_put = self.net(x)
        return out_put

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

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, train_bn=False, affine_par=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = train_bn

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = train_bn
        self.conv3 = nn.Conv2d(planes, planes * 1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 1, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = train_bn
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='up',
                 BN_enable=True, norm_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                                    kernel_size=3, stride=1, padding=0, bias=False))

        if self.BN_enable:
            self.norm1 = norm_layer(out_channels)
        self.relu1 = nn.ReLU(inplace=False)

        if self.upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,
                                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode == 'pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)
        elif self.upsample_mode == 'up':
            self.upsample = nn.Upsample(scale_factor=2)
        #if self.BN_enable:
        #    self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = self.relu1(x)
        x = self.upsample(x)
        return x

class ResNet(nn.Module):

    def __init__(self, block, layers, width=1.0, in_channel=3, unet=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        self.unet = unet
        self._norm_layer = nn.BatchNorm2d

        #self.inplanes = max(int(64 * width), 64)
        self.inplanes = int(64 * width)
        self.base = int(64 * width)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer0 = nn.Sequential(
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
            #x = self.avgpool(x)
            return x

class ResUnet(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, layer, BasicBlock, in_channel=4, width=1.0, BN_enable=True, pretrained=False):
        super(ResUnet, self).__init__()

        filter = [64, 64, 128, 256, 512]
        filters = [int(i * width) for i in filter]
        self.BN_enable = BN_enable
        self.pretrain = pretrained
        self.width = width
        # use split batchnorm
        norm_layer = nn.BatchNorm2d

        self.encoder = ResNet(BasicBlock, layer, width=width, in_channel=in_channel, unet=True, norm_layer=norm_layer)

        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3],
                                   BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1], BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder3 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[0] * 2, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ins = nn.Sequential(nn.Linear(filters[2], filters[1]), Normalize(2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, mode=0, recon=False):
        if mode == 0:
            feat = self.encoder(x, 0)
            e1 = feat['d1']
            e2 = feat['d3']
            e3 = feat['d4']
            e4 = feat['d5']
            ins = self.fc_ins(self.avgpool(e4).squeeze())
            center = self.center(e4)
            d2 = self.decoder1(torch.cat([center, e3], dim=1))
            d3 = self.decoder2(torch.cat([d2, e2], dim=1))
            d4 = self.decoder3(torch.cat([d3, e1], dim=1))
            return d4, ins
        else:
            feat = self.encoder(x, 0)
            e1 = feat['d1']
            e2 = feat['d3']
            e3 = feat['d4']
            e4 = feat['d5']
            center = self.center(e4)
            d2 = self.decoder1(torch.cat([center, e3], dim=1))
            d3 = self.decoder2(torch.cat([d2, e2], dim=1))
            d4 = self.decoder3(torch.cat([d3, e1], dim=1))
            return d4

class ResUnet2(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, layer, BasicBlock, in_channel=4, width=1.0, BN_enable=True, pretrained=False):
        super(ResUnet2, self).__init__()

        filter = [64, 64, 128, 256, 512]
        filters = [int(i * width) for i in filter]
        self.BN_enable = BN_enable
        self.pretrain = pretrained
        self.width = width
        # use split batchnorm
        norm_layer = nn.BatchNorm2d

        self.encoder = ResNet(BasicBlock, layer, width=width, in_channel=in_channel, unet=True, norm_layer=norm_layer)

        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3],
                                   BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable, upsample_mode='up',
                                     norm_layer=norm_layer)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[2] + filters[1], BN_enable=self.BN_enable, upsample_mode='up',
                                     norm_layer=norm_layer)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[3], BN_enable=self.BN_enable, upsample_mode='up',
                                     norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, mode=0, recon=False):
        if mode == 0:
            feat = self.encoder(x, 0)
            e1 = feat['d1']
            e2 = feat['d3']
            e3 = feat['d4']
            e4 = feat['d5']
            center = self.center(e4)
            d2 = self.decoder1(torch.cat([center, e3], dim=1))
            d3 = self.decoder2(torch.cat([d2, e2], dim=1))
            d4 = self.decoder3(torch.cat([d3, e1], dim=1))
            return d4, e4
        else:
            feat = self.encoder(x, 0)
            e1 = feat['d1']
            e2 = feat['d3']
            e3 = feat['d4']
            e4 = feat['d5']
            center = self.center(e4)
            d2 = self.decoder1(torch.cat([center, e3], dim=1))
            d3 = self.decoder2(torch.cat([d2, e2], dim=1))
            d4 = self.decoder3(torch.cat([d3, e1], dim=1))
            return d4

class ResYnet(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, layer, BasicBlock, in_channel=4, width=1.0, BN_enable=True, pretrained=False):
        super(ResYnet, self).__init__()

        filters = [64, 64, 128, 256, 512]
        self.BN_enable = BN_enable
        self.pretrain = pretrained
        self.width = width
        # use split batchnorm
        norm_layer = nn.BatchNorm2d

        self.encoder1 = ResNet(BasicBlock, layer, width=0.5, in_channel=4, unet=True, norm_layer=norm_layer)
        self.encoder2 = ResNet(BasicBlock, layer, width=0.5, in_channel=2, unet=True, norm_layer=norm_layer)

        # decoder部分
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3],
                                   BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1], BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.decoder3 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[1] * 2, BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ins1 = nn.Sequential(nn.Linear(filters[2], filters[1]), Normalize(2))
        self.fc_ins2 = nn.Sequential(nn.Linear(filters[2], filters[1]), Normalize(2))
        self.loss = HardNegtive_loss()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, mode=0):
        if mode == 0:
            x1, x2 = torch.split(x, [4, 2], dim=1)
            outputs1 = self.encoder1(x1, mode=0)
            outputs2 = self.encoder2(x2, mode=0)
            e11 = outputs1['d1']
            e21 = outputs2['d1']
            e1 = torch.cat((e11, e21), dim=1)
            e12 = outputs1['d3']
            e22 = outputs2['d3']
            e2 = torch.cat((e12, e22), dim=1)
            e13 = outputs1['d4']
            e23 = outputs2['d4']
            e3 = torch.cat((e13, e23), dim=1)
            e14 = outputs1['d5']
            e24 = outputs2['d5']
            e4 = torch.cat((e14, e24), dim=1)
            # ins feature
            ins1 = self.fc_ins1(self.avgpool(e14).squeeze())
            ins2 = self.fc_ins2(self.avgpool(e24).squeeze())
            loss = self.loss(ins1, ins2)
            center = self.center(e4)
            d2 = self.decoder1(torch.cat([center, e3], dim=1))
            d3 = self.decoder2(torch.cat([d2, e2], dim=1))
            d4 = self.decoder3(torch.cat([d3, e1], dim=1))
            return d4, e4, loss
        else:
            x1, x2 = torch.split(x, [4, 2], dim=1)
            outputs1 = self.encoder1(x1, mode=0)
            outputs2 = self.encoder2(x2, mode=0)
            e11 = outputs1['d1']
            e21 = outputs2['d1']
            e1 = torch.cat((e11, e21), dim=1)
            e12 = outputs1['d3']
            e22 = outputs2['d3']
            e2 = torch.cat((e12, e22), dim=1)
            e13 = outputs1['d4']
            e23 = outputs2['d4']
            e3 = torch.cat((e13, e23), dim=1)
            e14 = outputs1['d5']
            e24 = outputs2['d5']
            e4 = torch.cat((e14, e24), dim=1)
            center = self.center(e4)
            d2 = self.decoder1(torch.cat([center, e3], dim=1))
            d3 = self.decoder2(torch.cat([d2, e2], dim=1))
            d4 = self.decoder3(torch.cat([d3, e1], dim=1))
            return d4

class ResYnet2(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, layer, BasicBlock, in_channel=4, width=1.0, BN_enable=True, pretrained=False):
        super(ResYnet2, self).__init__()

        filters = [64, 64, 128, 256, 512]
        self.BN_enable = BN_enable
        self.pretrain = pretrained
        self.width = width
        # use split batchnorm
        norm_layer = nn.BatchNorm2d

        self.encoder1 = ResNet(BasicBlock, layer, width=0.5, in_channel=4, unet=True, norm_layer=norm_layer)
        self.encoder2 = ResNet(BasicBlock, layer, width=0.5, in_channel=2, unet=True, norm_layer=norm_layer)

        # decoder部分
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3],
                                   BN_enable=self.BN_enable, upsample_mode='up', norm_layer=norm_layer)
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable, upsample_mode='up',
                                     norm_layer=norm_layer)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=192, BN_enable=self.BN_enable, upsample_mode='up',
                                     norm_layer=norm_layer)
        self.decoder3 = DecoderBlock(in_channels=192 + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[3], BN_enable=self.BN_enable, upsample_mode='up',
                                     norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ins1 = nn.Sequential(nn.Linear(filters[2], filters[1]), Normalize(2))
        self.fc_ins2 = nn.Sequential(nn.Linear(filters[2], filters[1]), Normalize(2))
        self.loss = HardNegtive_loss()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, mode=0):
        if mode == 0:
            x1, x2 = torch.split(x, [4, 2], dim=1)
            outputs1 = self.encoder1(x1, mode=0)
            outputs2 = self.encoder2(x2, mode=0)
            e11 = outputs1['d1']
            e21 = outputs2['d1']
            e1 = torch.cat((e11, e21), dim=1)
            e12 = outputs1['d3']
            e22 = outputs2['d3']
            e2 = torch.cat((e12, e22), dim=1)
            e13 = outputs1['d4']
            e23 = outputs2['d4']
            e3 = torch.cat((e13, e23), dim=1)
            e14 = outputs1['d5']
            e24 = outputs2['d5']
            e4 = torch.cat((e14, e24), dim=1)
            # ins feature
            ins1 = self.fc_ins1(self.avgpool(e14).squeeze())
            ins2 = self.fc_ins2(self.avgpool(e24).squeeze())
            loss = self.loss(ins1, ins2)
            center = self.center(e4)
            d2 = self.decoder1(torch.cat([center, e3], dim=1))
            d3 = self.decoder2(torch.cat([d2, e2], dim=1))
            d4 = self.decoder3(torch.cat([d3, e1], dim=1))
            return d4, e4, loss
        else:
            x1, x2 = torch.split(x, [4, 2], dim=1)
            outputs1 = self.encoder1(x1, mode=0)
            outputs2 = self.encoder2(x2, mode=0)
            e11 = outputs1['d1']
            e21 = outputs2['d1']
            e1 = torch.cat((e11, e21), dim=1)
            e12 = outputs1['d3']
            e22 = outputs2['d3']
            e2 = torch.cat((e12, e22), dim=1)
            e13 = outputs1['d4']
            e23 = outputs2['d4']
            e3 = torch.cat((e13, e23), dim=1)
            e14 = outputs1['d5']
            e24 = outputs2['d5']
            e4 = torch.cat((e14, e24), dim=1)
            center = self.center(e4)
            d2 = self.decoder1(torch.cat([center, e3], dim=1))
            d3 = self.decoder2(torch.cat([d2, e2], dim=1))
            d4 = self.decoder3(torch.cat([d3, e1], dim=1))
            return d4

class M_Fusion(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, width=1, in_channel=4, in_dim=128, feat_dim=128):
        super(M_Fusion, self).__init__()
        # use split batchnorm
        self.encoder = ResYnet2([2, 2, 2, 2], BasicBlock, width=width, in_channel=6)
        # use split
        self.norm = Normalize(2)
        self.fc = nn.Sequential(nn.Linear(256, 128), Normalize(2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ins1 = nn.Conv2d(in_dim, int(0.5 * feat_dim), kernel_size=1, stride=1)
        self.loss = HardNegtive_loss()

    def forward(self, x1, x2, mode=0):
        # mode --
        if mode == 0:
            # local
            feat1, ins1, floss1 = self.encoder(x1)
            feat2, ins2, floss2 = self.encoder(x2)
            feat1 = self.fc_ins1(feat1)
            feat1 = self.norm(feat1)
            feat2 = self.fc_ins1(feat2)
            feat2 = self.norm(feat2)
            # global
            ins1 = self.fc(self.avgpool(ins1).squeeze())
            ins2 = self.fc(self.avgpool(ins2).squeeze())
            loss = self.loss(ins1, ins2)
            return feat1, feat2, loss + floss1 + floss2
        else:
            fet1 = self.encoder(x1, mode=1)
            return fet1


class L_Fusion(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, width=0.5, in_channel=4, in_dim=128, feat_dim=128):
        super(L_Fusion, self).__init__()
        # use split batchnorm
        self.encoder1 = ResUnet2([2, 2, 2, 2], BasicBlock, width=width, in_channel=4)
        self.encoder2 = ResUnet2([2, 2, 2, 2], BasicBlock, width=width, in_channel=2)
        # use split
        self.norm = Normalize(2)
        self.fc = nn.Sequential(nn.Linear(128, 64), Normalize(2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ins1 = nn.Conv2d(in_dim, int(0.5 * feat_dim), kernel_size=1, stride=1)
        self.fc_ins2 = nn.Conv2d(in_dim, int(0.5 * feat_dim), kernel_size=1, stride=1)
        self.loss = HardNegtive_loss()


    def forward(self, x1, x2, mode=0):
        # mode --
        if mode == 0:
            feat1, ins1 = self.encoder1(x1)
            feat2, ins2 = self.encoder2(x2)
            # image-level
            ins1 = self.fc(self.avgpool(ins1).squeeze())
            ins2 = self.fc(self.avgpool(ins2).squeeze())
            loss = self.loss(ins1, ins2)
            # normalization
            feat1 = self.fc_ins1(feat1)
            feat1 = self.norm(feat1)
            feat2 = self.fc_ins2(feat2)
            feat2 = self.norm(feat2)
            return feat1, feat2, loss
        else:
            feat1 = self.encoder1(x1, mode=1)
            feat2 = self.encoder2(x2, mode=1)
            return feat1, feat2

class E_Fusion(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, width=1, in_channel=4, in_dim=128, feat_dim=128):
        super(E_Fusion, self).__init__()
        # use split batchnorm
        self.encoder = ResUnet2([2, 2, 2, 2], BasicBlock, width=width, in_channel=in_channel)
        # use split
        self.norm = Normalize(2)
        self.fc = nn.Sequential(nn.Linear(256, 128), Normalize(2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ins1 = nn.Conv2d(in_dim, int(0.5 * feat_dim), kernel_size=1, stride=1)
        self.loss = HardNegtive_loss()


    def forward(self, x1, x2, mode=0):
        # mode --
        if mode == 0:
            feat1, ins1 = self.encoder(x1)
            feat2, ins2 = self.encoder(x2)
            # image-level
            ins1 = self.fc(self.avgpool(ins1).squeeze())
            ins2 = self.fc(self.avgpool(ins2).squeeze())
            loss = self.loss(ins1, ins2)
            # normalization
            feat1 = self.fc_ins1(feat1)
            feat1 = self.norm(feat1)
            feat2 = self.fc_ins1(feat2)
            feat2 = self.norm(feat2)
            return feat1, feat2, loss
        else:
            feat = self.encoder(x1, mode=1)
            return feat
