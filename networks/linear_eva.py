import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out_put = self.net(x)
        return out_put


class ResMlp(nn.Module):

    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(ResMlp, self).__init__()

        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=4)
        self.up3 = nn.Upsample(scale_factor=8)
        self.up4 = nn.Upsample(scale_factor=16)

        self.classifier = nn.Conv2d(in_channels, projection_size, kernel_size=1, stride=1)

    def forward(self, outputs1, outputs2):

        e11 = outputs1['d1'].detach()
        e21 = outputs2['d1'].detach()
        e1 = self.up1(torch.cat((e11, e21), dim=1))
        e12 = outputs1['d3'].detach()
        e22 = outputs2['d3'].detach()
        e2 = self.up2(torch.cat((e12, e22), dim=1))
        e13 = outputs1['d4'].detach()
        e23 = outputs2['d4'].detach()
        e3 = self.up3(torch.cat((e13, e23), dim=1))
        e14 = outputs1['d5'].detach()
        e24 = outputs2['d5'].detach()
        e4 = self.up4(torch.cat((e14, e24), dim=1))

        FM = torch.cat([e1, e2, e3, e4], dim=1)
        out_put = self.classifier(FM)

        return out_put

class ResMlpS(nn.Module):

    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(ResMlpS, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=4)
        self.up3 = nn.Upsample(scale_factor=8)
        self.up4 = nn.Upsample(scale_factor=16)

        self.classifier = nn.Conv2d(in_channels, projection_size, kernel_size=1, stride=1)

    def forward(self, outputs):
        e1 = outputs['d1'].detach()
        e1 = self.up1(e1)
        e2 = outputs['d3'].detach()
        e2 = self.up2(e2)
        e3 = outputs['d4'].detach()
        e3 = self.up3(e3)
        e4 = outputs['d5'].detach()
        e4 = self.up4(e4)

        FM = torch.cat([e1, e2, e3, e4], dim=1)
        out_put = self.classifier(FM)

        return out_put