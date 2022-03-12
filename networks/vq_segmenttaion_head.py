import math
import torch
import torch.nn as nn
from torch import einsum
from torch.nn import init
from einops import rearrange
import torch.nn.functional as F


__all__ = ['ResNet']

class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

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
            return x


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
                                     out_channels=filters[2], BN_enable=self.BN_enable, upsample_mode='up',
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



class E_Fusion(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, width=1, in_channel=6, in_dim=128, feat_dim=128):
        super(E_Fusion, self).__init__()
        # parameters
        self.temperature = 0.8
        self.kld_scale = 5e-4
        # use split batchnorm
        self.encoder = ResUnet2([2, 2, 2, 2], BasicBlock, width=width, in_channel=in_channel)
        # use split
        self.norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.fc5 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.fc6 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.fc_unet = nn.Conv2d(in_dim, feat_dim, kernel_size=1, stride=1)
        self.embed1 = nn.Embedding(1, 128)
        self.embed2 = nn.Embedding(1, 128)
        self.embed3 = nn.Embedding(1, 128)
        self.embed4 = nn.Embedding(1, 128)
        self.embed5 = nn.Embedding(1, 128)
        self.embed6 = nn.Embedding(1, 128)


    def forward(self, x, bsz, mode=0):
        temp = self.temperature
        # mode --
        if mode == 0:
            feat, ins = self.encoder(x)
            feat = self.fc_unet(feat)
            feat1 = feat
            # image-level
            gns1 = self.fc1(self.avgpool(ins).squeeze()).unsqueeze(dim=2)
            gns2 = self.fc2(self.avgpool(ins).squeeze()).unsqueeze(dim=2)
            gns3 = self.fc3(self.avgpool(ins).squeeze()).unsqueeze(dim=2)
            gns4 = self.fc4(self.avgpool(ins).squeeze()).unsqueeze(dim=2)
            gns5 = self.fc5(self.avgpool(ins).squeeze()).unsqueeze(dim=2)
            gns6 = self.fc6(self.avgpool(ins).squeeze()).unsqueeze(dim=2)
            ins1 = self.embed1.weight.expand(bsz, 128).unsqueeze(dim=2)
            ins2 = self.embed2.weight.expand(bsz, 128).unsqueeze(dim=2)
            ins3 = self.embed3.weight.expand(bsz, 128).unsqueeze(dim=2)
            ins4 = self.embed4.weight.expand(bsz, 128).unsqueeze(dim=2)
            ins5 = self.embed5.weight.expand(bsz, 128).unsqueeze(dim=2)
            ins6 = self.embed6.weight.expand(bsz, 128).unsqueeze(dim=2)
            embd = torch.cat([ins1, ins2, ins3, ins4, ins5, ins6], dim=-1)
            glob = torch.cat([gns1, gns2, gns3, gns4, gns5, gns6], dim=-1)
            divs = torch.cat([self.embed1.weight, self.embed2.weight, self.embed3.weight,
                              self.embed4.weight, self.embed5.weight, self.embed6.weight], dim=0)
            # align and scatter loss
            loss_bt = self.align_loss(embd, glob)
            loss_dv = self.uniform_loss(self.norm(divs))
            # normalization
            feat = rearrange(feat, 'b c h w -> b h w c')
            fb, fm, fn, fc = feat.shape
            # Flatten feat
            feat = feat.contiguous().view(fb, -1, fc)
            proj = einsum('ijk,ikl -> ijl', feat, embd)
            soft_one_hot = F.gumbel_softmax(proj, tau=temp, dim=-1, hard=False)
            # diversity loss + kl divergence to the prior loss
            qy = F.softmax(proj, dim=1)
            diff = self.kld_scale * torch.sum(qy * torch.log(qy * 6 + 1e-10), dim=1).mean()
            # pixel-wise feature
            embd = rearrange(embd, 'i k l -> i l k')
            feat2 = einsum('ijl, ilk -> ijk', soft_one_hot, embd)
            feat2 = feat2.contiguous().view(fb, fm, fn, fc)
            feat2 = rearrange(feat2, 'b h w c -> b c h w')
            # normalization
            feat1 = self.norm(feat1)
            feat2 = self.norm(feat2)
            return feat1, feat2, diff + loss_bt + loss_dv

        else:
            feat, ins = self.encoder(x)
            feat = self.fc_unet(feat)
            # image-level
            ins1 = self.embed1.weight.expand(bsz, 128).unsqueeze(dim=2)
            ins2 = self.embed2.weight.expand(bsz, 128).unsqueeze(dim=2)
            ins3 = self.embed3.weight.expand(bsz, 128).unsqueeze(dim=2)
            ins4 = self.embed4.weight.expand(bsz, 128).unsqueeze(dim=2)
            ins5 = self.embed5.weight.expand(bsz, 128).unsqueeze(dim=2)
            ins6 = self.embed6.weight.expand(bsz, 128).unsqueeze(dim=2)
            embd = torch.cat([ins1, ins2, ins3, ins4, ins5, ins6], dim=-1)
            # normalization
            feat = rearrange(feat, 'b c h w -> b h w c')
            fb, fm, fn, fc = feat.shape
            # Flatten feat
            feat = feat.contiguous().view(fb, -1, fc)
            proj = einsum('ijk,ikl -> ijl', feat, embd)
            soft_one_hot = F.gumbel_softmax(proj, tau=self.temperature, dim=-1, hard=False)
            soft_one_hot = soft_one_hot.contiguous().view(fb, fm, fn, -1)
            soft_one_hot = rearrange(soft_one_hot, 'b h w c -> b c h w')
            soft_one_hot = torch.argmax(soft_one_hot, dim=1)
            '''
            embd = rearrange(embd, 'i k l -> i l k')
            feat2 = einsum('ijl, ilk -> ijk', soft_one_hot, embd)
            feat2 = feat2.contiguous().view(fb, fm, fn, fc)
            feat2 = rearrange(feat2, 'b h w c -> b c h w')
            '''
            return soft_one_hot

    def BarlowTwinsLoss(self, z1: torch.Tensor, z2: torch.Tensor):
        # normalize repr. along the batch dimension
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)  # NxD
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)  # NxD
        z_1 = z1_norm.permute(1, 0, 2)  # 128, b, 6
        z_2 = z2_norm.permute(1, 2, 0)  # 128, 6, b

        B = z_1.size(0)
        N = z_1.size(1)
        D = z_1.size(2)

        # cross-correlation matrix
        c = torch.bmm(z_2, z_1) / N  # 128 x 6 x 6
        one = torch.eye(D, device=z1.device).unsqueeze(0).repeat(B, 1, 1) # 128x6x6
        # loss
        c_diff = (c - one).pow(2)  # 128 x 6 x 6
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~one.bool()] *= 1 / D
        loss = c_diff.sum() / B
        return loss

    def align_loss(sef, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class M_Segment(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, width=1, classnum=6, in_dim=128, feat_dim=128):
        super(M_Segment, self).__init__()
        # use split batchnorm
        self.encoder = ResUnet2([2, 2, 2, 2], BasicBlock, width=width, in_channel=6)
        self.fc = nn.Conv2d(feat_dim, classnum, kernel_size=1, stride=1)

    def forward(self, x1, mode=0):
        # mode --
        out = self.encoder(x1, mode=1)
        out = self.fc(out)
        return out