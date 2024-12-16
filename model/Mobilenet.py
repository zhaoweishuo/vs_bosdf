from torchvision import models
from torch import nn
import torch
from torchsummary import summary


class MobileNet3(nn.Module):

    def __init__(self, pretrained=False):
        super(MobileNet3, self).__init__()
        mobilenet = models.mobilenet_v3_small()
        if pretrained:
            mobilenet.load_state_dict(torch.load("./pretrained/mobilenet_v3_small-047dcff4.pth"))

        self.mobilenet = mobilenet

    def forward(self, x):
        y = self.mobilenet(x)
        return y


class Out(nn.Module):
    """Encoder for network"""
    def __init__(self):
        super(Out, self).__init__()

        self.layer1 = nn.Linear(1000, 500)
        self.layer2 = nn.Linear(500, 100)
        self.layer3 = nn.Linear(100, 6)

    def forward(self, x):
        out500 = self.layer1(x)
        out100 = self.layer2(out500)
        out6 = self.layer3(out100)
        return out6


class BODFNet(nn.Module):
    """Encoder for network"""
    def __init__(self, pretrained=False):
        super(BODFNet, self).__init__()

        self.mobilenet = MobileNet3(pretrained=pretrained)
        self.out = Out()

    def forward(self, x):
        out1000 = self.mobilenet(x)
        out6 = self.out(out1000)
        return out1000, out6
